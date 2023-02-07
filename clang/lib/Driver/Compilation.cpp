//===- Compilation.cpp - Compilation Task Implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

using namespace clang;
using namespace driver;
using namespace llvm::opt;

Compilation::Compilation(const Driver &D, const ToolChain &_DefaultToolChain,
                         InputArgList *_Args, DerivedArgList *_TranslatedArgs,
                         bool ContainsError)
    : TheDriver(D), DefaultToolChain(_DefaultToolChain), Args(_Args),
      TranslatedArgs(_TranslatedArgs), ContainsError(ContainsError) {
  // The offloading host toolchain is the default toolchain.
  OrderedOffloadingToolchains.insert(
      std::make_pair(Action::OFK_Host, &DefaultToolChain));
}

Compilation::~Compilation() {
  // Remove temporary files. This must be done before arguments are freed, as
  // the file names might be derived from the input arguments.
  if (!TheDriver.isSaveTempsEnabled() && !ForceKeepTempFiles)
    CleanupFileList(TempFiles);

  delete TranslatedArgs;
  delete Args;

  // Free any derived arg lists.
  for (auto Arg : TCArgs)
    if (Arg.second != TranslatedArgs)
      delete Arg.second;
}

const DerivedArgList &
Compilation::getArgsForToolChain(const ToolChain *TC, StringRef BoundArch,
                                 Action::OffloadKind DeviceOffloadKind) {
  if (!TC)
    TC = &DefaultToolChain;

  DerivedArgList *&Entry = TCArgs[{TC, BoundArch, DeviceOffloadKind}];
  if (!Entry) {
    SmallVector<Arg *, 4> AllocatedArgs;
    DerivedArgList *OpenMPArgs = nullptr;
    // Translate OpenMP toolchain arguments provided via the -Xopenmp-target flags.
    if (DeviceOffloadKind == Action::OFK_OpenMP) {
      const ToolChain *HostTC = getSingleOffloadToolChain<Action::OFK_Host>();
      bool SameTripleAsHost = (TC->getTriple() == HostTC->getTriple());
      OpenMPArgs = TC->TranslateOpenMPTargetArgs(
          *TranslatedArgs, SameTripleAsHost, AllocatedArgs);
    }

    DerivedArgList *NewDAL = nullptr;
    if (!OpenMPArgs) {
      NewDAL = TC->TranslateXarchArgs(*TranslatedArgs, BoundArch,
                                      DeviceOffloadKind, &AllocatedArgs);
    } else {
      NewDAL = TC->TranslateXarchArgs(*OpenMPArgs, BoundArch, DeviceOffloadKind,
                                      &AllocatedArgs);
      if (!NewDAL)
        NewDAL = OpenMPArgs;
      else
        delete OpenMPArgs;
    }

    if (!NewDAL) {
      Entry = TC->TranslateArgs(*TranslatedArgs, BoundArch, DeviceOffloadKind);
      if (!Entry)
        Entry = TranslatedArgs;
    } else {
      Entry = TC->TranslateArgs(*NewDAL, BoundArch, DeviceOffloadKind);
      if (!Entry)
        Entry = NewDAL;
      else
        delete NewDAL;
    }

    // Add allocated arguments to the final DAL.
    for (auto *ArgPtr : AllocatedArgs)
      Entry->AddSynthesizedArg(ArgPtr);
  }

  return *Entry;
}

bool Compilation::CleanupFile(const char *File, bool IssueErrors) const {
  // FIXME: Why are we trying to remove files that we have not created? For
  // example we should only try to remove a temporary assembly file if
  // "clang -cc1" succeed in writing it. Was this a workaround for when
  // clang was writing directly to a .s file and sometimes leaving it behind
  // during a failure?

  // FIXME: If this is necessary, we can still try to split
  // llvm::sys::fs::remove into a removeFile and a removeDir and avoid the
  // duplicated stat from is_regular_file.

  // Don't try to remove files which we don't have write access to (but may be
  // able to remove), or non-regular files. Underlying tools may have
  // intentionally not overwritten them.
  if (!llvm::sys::fs::can_write(File) || !llvm::sys::fs::is_regular_file(File))
    return true;

  if (std::error_code EC = llvm::sys::fs::remove(File)) {
    // Failure is only failure if the file exists and is "regular". We checked
    // for it being regular before, and llvm::sys::fs::remove ignores ENOENT,
    // so we don't need to check again.

    if (IssueErrors)
      getDriver().Diag(diag::err_drv_unable_to_remove_file)
        << EC.message();
    return false;
  }
  return true;
}

bool Compilation::CleanupFileList(const llvm::opt::ArgStringList &Files,
                                  bool IssueErrors) const {
  bool Success = true;
  for (const auto &File: Files)
    Success &= CleanupFile(File, IssueErrors);
  return Success;
}

bool Compilation::CleanupFileMap(const ArgStringMap &Files,
                                 const JobAction *JA,
                                 bool IssueErrors) const {
  bool Success = true;
  for (const auto &File : Files) {
    // If specified, only delete the files associated with the JobAction.
    // Otherwise, delete all files in the map.
    if (JA && File.first != JA)
      continue;
    Success &= CleanupFile(File.second, IssueErrors);
  }
  return Success;
}

int Compilation::ExecuteCommand(const Command &C,
                                const Command *&FailingCommand,
                                bool LogOnly) const {
  if ((getDriver().CCPrintOptions ||
       getArgs().hasArg(options::OPT_v)) && !getDriver().CCGenDiagnostics) {
    raw_ostream *OS = &llvm::errs();
    std::unique_ptr<llvm::raw_fd_ostream> OwnedStream;

    // Follow gcc implementation of CC_PRINT_OPTIONS; we could also cache the
    // output stream.
    if (getDriver().CCPrintOptions &&
        !getDriver().CCPrintOptionsFilename.empty()) {
      std::error_code EC;
      OwnedStream.reset(new llvm::raw_fd_ostream(
          getDriver().CCPrintOptionsFilename, EC,
          llvm::sys::fs::OF_Append | llvm::sys::fs::OF_TextWithCRLF));
      if (EC) {
        getDriver().Diag(diag::err_drv_cc_print_options_failure)
            << EC.message();
        FailingCommand = &C;
        return 1;
      }
      OS = OwnedStream.get();
    }

    if (getDriver().CCPrintOptions)
      *OS << "[Logging clang options]\n";

    C.Print(*OS, "\n", /*Quote=*/getDriver().CCPrintOptions);
  }

  if (LogOnly)
    return 0;

  std::string Error;
  bool ExecutionFailed;
  int Res = C.Execute(Redirects, &Error, &ExecutionFailed);
  if (PostCallback)
    PostCallback(C, Res);
  if (!Error.empty()) {
    assert(Res && "Error string set with 0 result code!");
    getDriver().Diag(diag::err_drv_command_failure) << Error;
  }

  if (Res)
    FailingCommand = &C;

  return ExecutionFailed ? 1 : Res;
}

using FailingCommandList = SmallVectorImpl<std::pair<int, const Command *>>;

static bool ActionFailed(const Action *A,
                         const FailingCommandList &FailingCommands) {
  if (FailingCommands.empty())
    return false;

  // CUDA/HIP can have the same input source code compiled multiple times so do
  // not compiled again if there are already failures. It is OK to abort the
  // CUDA pipeline on errors.
  if (A->isOffloading(Action::OFK_Cuda) || A->isOffloading(Action::OFK_HIP))
    return true;

  for (const auto &CI : FailingCommands)
    if (A == &(CI.second->getSource()))
      return true;

  for (const auto *AI : A->inputs())
    if (ActionFailed(AI, FailingCommands))
      return true;

  return false;
}

static bool InputsOk(const Command &C,
                     const FailingCommandList &FailingCommands) {
  return !ActionFailed(&C.getSource(), FailingCommands);
}

namespace {
class JobScheduler {
public:
  enum JobState { JS_WAIT, JS_RUN, JS_DONE, JS_FAIL };
  JobScheduler(const JobList &Jobs, size_t NJobs = 1)
      : Jobs(Jobs), NumJobs(NJobs) {
#if !LLVM_ENABLE_THREADS
    NumJobs = 1;
#endif
    for (auto &Job : Jobs) {
      JState[&Job] = JS_WAIT;
      for (const auto *AI : Job.getDependentActions()) {
        for (const auto *CI : ActToCmds[AI]) {
          DependentCmds[&Job].push_back(CI);
        }
      }
      for (const auto *CI : ActToCmds[&Job.getSource()]) {
        DependentCmds[&Job].push_back(CI);
      }
      ActToCmds[&Job.getSource()].push_back(&Job);
    }
  }
  /// \return true if all jobs are done. Otherwise, \p Next contains the
  /// the next job ready to be executed if it is not null pointer. Otherwise
  /// all jobs are running or waiting.
  bool IsDone(const Command *&Next) {
    std::lock_guard<std::mutex> lock(Mutex);
    Next = nullptr;
    unsigned Done = 0;
    unsigned Running = 0;
    for (auto &Cmd : Jobs) {
      switch (JState[&Cmd]) {
      case JS_RUN:
        ++Running;
        break;
      case JS_DONE:
      case JS_FAIL:
        ++Done;
        break;
      case JS_WAIT: {
        bool InputsReady = true;
        for (const auto *CI : DependentCmds[&Cmd]) {
          if (JState[CI] == JS_FAIL) {
            JState[&Cmd] = JS_FAIL;
            ++Done;
            InputsReady = false;
            break;
          }
          if (JState[CI] != JS_DONE) {
            InputsReady = false;
            break;
          }
        }
        if (!Next && InputsReady) {
          Next = &Cmd;
        }
        break;
      }
      }
    }
    if (Running >= NumJobs)
      Next = nullptr;
    return Done == Jobs.size();
  }

  void setJobState(const Command *Cmd, JobState JS) {
    std::lock_guard<std::mutex> lock(Mutex);
    JState[Cmd] = JS;
  }

  void launch(std::function<void()> Work) {
#if LLVM_ENABLE_THREADS
    if (NumJobs == 1) {
      Work();
      return;
    }
    std::thread Th(Work);
    Th.detach();
#else
    Work();
#endif
  }

private:
  std::mutex Mutex;
  const JobList &Jobs;
  llvm::DenseMap<const Command *, JobState> JState;
  llvm::DenseMap<const Action *, llvm::SmallVector<const Command *, 4>>
      ActToCmds;
  llvm::DenseMap<const Command *, llvm::SmallVector<const Command *, 4>>
      DependentCmds;
  size_t NumJobs; // Number of parallel jobs to run
};
} // namespace
void Compilation::ExecuteJobs(const JobList &Jobs,
                              FailingCommandList &FailingCommands,
                              bool LogOnly) const {
  // According to UNIX standard, driver need to continue compiling all the
  // inputs on the command line even one of them failed.
  // In all but CLMode, execute all the jobs unless the necessary inputs for the
  // job is missing due to previous failures.
  JobScheduler JS(Jobs, getDriver().getNumberOfParallelJobs());

  const Command *Next = nullptr;
  while (!JS.IsDone(Next)) {
    if (!Next) {
      // sleep, rather than yield so we do not busy wait.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    if (!InputsOk(*Next, FailingCommands)) {
      JS.setJobState(Next, JobScheduler::JS_FAIL);
      // Bail as soon as one command fails in cl driver mode.
      if (TheDriver.IsCLMode())
        return;
      continue;
    }

    JS.setJobState(Next, JobScheduler::JS_RUN);
    auto Work = [&, Next]() {
      const Command *FailingCommand = nullptr;
      if (int Res = ExecuteCommand(*Next, FailingCommand, LogOnly)) {
        FailingCommands.push_back(std::make_pair(Res, FailingCommand));
        JS.setJobState(Next, JobScheduler::JS_FAIL);
      } else {
        JS.setJobState(Next, JobScheduler::JS_DONE);
      }
    };
    JS.launch(Work);
  }
}

void Compilation::initCompilationForDiagnostics() {
  ForDiagnostics = true;

  // Free actions and jobs.
  Actions.clear();
  AllActions.clear();
  Jobs.clear();

  // Remove temporary files.
  if (!TheDriver.isSaveTempsEnabled() && !ForceKeepTempFiles)
    CleanupFileList(TempFiles);

  // Clear temporary/results file lists.
  TempFiles.clear();
  ResultFiles.clear();
  FailureResultFiles.clear();

  // Remove any user specified output.  Claim any unclaimed arguments, so as
  // to avoid emitting warnings about unused args.
  OptSpecifier OutputOpts[] = {
      options::OPT_o,  options::OPT_MD, options::OPT_MMD, options::OPT_M,
      options::OPT_MM, options::OPT_MF, options::OPT_MG,  options::OPT_MJ,
      options::OPT_MQ, options::OPT_MT, options::OPT_MV};
  for (const auto &Opt : OutputOpts) {
    if (TranslatedArgs->hasArg(Opt))
      TranslatedArgs->eraseArg(Opt);
  }
  TranslatedArgs->ClaimAllArgs();

  // Force re-creation of the toolchain Args, otherwise our modifications just
  // above will have no effect.
  for (auto Arg : TCArgs)
    if (Arg.second != TranslatedArgs)
      delete Arg.second;
  TCArgs.clear();

  // Redirect stdout/stderr to /dev/null.
  Redirects = {std::nullopt, {""}, {""}};

  // Temporary files added by diagnostics should be kept.
  ForceKeepTempFiles = true;
}

StringRef Compilation::getSysRoot() const {
  return getDriver().SysRoot;
}

void Compilation::Redirect(ArrayRef<std::optional<StringRef>> Redirects) {
  this->Redirects = Redirects;
}
