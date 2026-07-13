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
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"
#include "clang/Options/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <system_error>
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
Compilation::getArgsForToolChain(const ToolChain *TC, BoundArch BA,
                                 Action::OffloadKind DeviceOffloadKind) {
  if (!TC)
    TC = &DefaultToolChain;

  DerivedArgList *&Entry = TCArgs[{TC, BA, DeviceOffloadKind}];
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
      NewDAL = TC->TranslateXarchArgs(*TranslatedArgs, BA, DeviceOffloadKind,
                                      &AllocatedArgs);
    } else {
      NewDAL = TC->TranslateXarchArgs(*OpenMPArgs, BA, DeviceOffloadKind,
                                      &AllocatedArgs);
      if (!NewDAL)
        NewDAL = OpenMPArgs;
      else
        delete OpenMPArgs;
    }

    if (!NewDAL) {
      Entry = TC->TranslateArgs(*TranslatedArgs, BA, DeviceOffloadKind);
      if (!Entry)
        Entry = TranslatedArgs;
    } else {
      Entry = TC->TranslateArgs(*NewDAL, BA, DeviceOffloadKind);
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

  // CUDA/HIP/SYCL can have the same input source code compiled multiple times
  // so do not compile again if there are already failures. It is OK to abort
  // the CUDA/HIP/SYCL pipeline on errors.
  if (A->isOffloading(Action::OFK_Cuda) || A->isOffloading(Action::OFK_HIP) ||
      A->isOffloading(Action::OFK_SYCL))
    return true;

  for (const auto &CI : FailingCommands)
    if (A == &(CI.second->getSource()))
      return true;

  for (const auto *AI : A->inputs())
    if (ActionFailed(AI, FailingCommands))
      return true;

  return false;
}

static bool ActionDependsOn(const Action *A, const Action *Other) {
  return A == Other || llvm::any_of(A->inputs(), [&](const Action *Input) {
           return ActionDependsOn(Input, Other);
         });
}

static bool ActionsAreIndependent(const Action *A, const Action *B) {
  return !ActionDependsOn(A, B) && !ActionDependsOn(B, A);
}

static bool CanRunInParallelOffloadJobGroup(const Command &Job) {
  return !Job.InProcess && !Job.PrintInputFilenames &&
         !Job.getBoundArch().empty() &&
         !Job.getOffloadDeviceParallelJobGroup().empty();
}

static bool SameParallelOffloadJobGroup(const Command &A, const Command &B) {
  return A.getOffloadDeviceParallelJobGroup() ==
         B.getOffloadDeviceParallelJobGroup();
}

static bool HasDistinctBoundArch(const Command &Candidate,
                                 ArrayRef<const Command *> Jobs) {
  BoundArch CandidateArch = Candidate.getBoundArch();
  return llvm::none_of(Jobs, [&](const Command *Job) {
    return Job->getBoundArch() == CandidateArch;
  });
}

static std::optional<llvm::ThreadPoolStrategy>
getParallelOffloadJobsStrategy(const ArgList &Args, unsigned NumJobs) {
  if (NumJobs < 2)
    return std::nullopt;

  auto OffloadJobs = tools::parseOffloadJobs(Args);
  if (!OffloadJobs.isValid())
    return std::nullopt;

  if (OffloadJobs.K == tools::OffloadJobsOpt::Kind::Jobserver)
    return llvm::jobserver_concurrency();

  if (OffloadJobs.NumThreads < 2)
    return std::nullopt;

  return llvm::hardware_concurrency(std::min(OffloadJobs.NumThreads, NumJobs));
}

struct ParallelJobResult {
  int Res = 0;
  bool ExecutionFailed = false;
  std::string Error;
};

struct ParallelOffloadJobGroupResult {
  size_t NumJobs = 0;
};

static std::optional<ParallelOffloadJobGroupResult>
tryExecuteParallelOffloadJobGroup(const Driver &D, const ArgList &Args,
                                  ArrayRef<std::optional<StringRef>> Redirects,
                                  const JobList::list_type &JobStorage,
                                  size_t StartIndex,
                                  FailingCommandList &FailingCommands) {
  const Command &Job = *JobStorage[StartIndex];
  if (!CanRunInParallelOffloadJobGroup(Job))
    return std::nullopt;

  SmallVector<const Command *, 4> ParallelJobs;
  for (size_t I = StartIndex; I < JobStorage.size(); ++I) {
    const Command &Candidate = *JobStorage[I];
    if (ActionFailed(&Candidate.getSource(), FailingCommands))
      break;

    if (!CanRunInParallelOffloadJobGroup(Candidate))
      break;

    if (!SameParallelOffloadJobGroup(Job, Candidate))
      break;

    if (!HasDistinctBoundArch(Candidate, ParallelJobs))
      break;

    if (!llvm::all_of(ParallelJobs, [&](const Command *Other) {
          return ActionsAreIndependent(&Candidate.getSource(),
                                       &Other->getSource());
        }))
      break;

    ParallelJobs.push_back(&Candidate);
  }

  std::optional<llvm::ThreadPoolStrategy> Strategy =
      getParallelOffloadJobsStrategy(Args, ParallelJobs.size());
  if (!Strategy)
    return std::nullopt;

  SmallVector<ParallelJobResult, 4> Results(ParallelJobs.size());
  llvm::DefaultThreadPool Pool(*Strategy);
  for (auto IndexedJob : llvm::enumerate(ParallelJobs)) {
    size_t Index = IndexedJob.index();
    const Command *ParallelJob = IndexedJob.value();
    Pool.async([&, Index, ParallelJob] {
      Results[Index].Res = ParallelJob->Execute(
          Redirects, &Results[Index].Error, &Results[Index].ExecutionFailed);
    });
  }
  Pool.wait();

  for (auto [Index, ParallelJob] : llvm::enumerate(ParallelJobs)) {
    ParallelJobResult &Result = Results[Index];
    if (!Result.Error.empty()) {
      assert(Result.Res && "Error string set with 0 result code!");
      D.Diag(diag::err_drv_command_failure) << Result.Error;
    }

    if (Result.Res) {
      FailingCommands.push_back(
          std::make_pair(Result.ExecutionFailed ? 1 : Result.Res, ParallelJob));
    }
  }

  return ParallelOffloadJobGroupResult{ParallelJobs.size()};
}

void Compilation::ExecuteJobs(const JobList &Jobs,
                              FailingCommandList &FailingCommands,
                              bool LogOnly) const {
  // According to UNIX standard, driver need to continue compiling all the
  // inputs on the command line even one of them failed.
  // In all but CLMode, execute all the jobs unless the necessary inputs for the
  // job is missing due to previous failures.
  bool CanRunJobsInParallel =
      !LogOnly && !getDriver().CCPrintOptions &&
      !getDriver().CCPrintProcessStats && !getDriver().CCGenDiagnostics &&
      !getDriver().IsCLMode() && !getArgs().hasArg(options::OPT_v) &&
      Redirects.empty() && !PostCallback;

  const auto &JobStorage = Jobs.getJobs();
  for (size_t I = 0; I < JobStorage.size();) {
    const auto &Job = *JobStorage[I];
    if (ActionFailed(&Job.getSource(), FailingCommands)) {
      ++I;
      continue;
    }

    if (CanRunJobsInParallel) {
      if (std::optional<ParallelOffloadJobGroupResult> Result =
              tryExecuteParallelOffloadJobGroup(getDriver(), getArgs(),
                                                Redirects, JobStorage, I,
                                                FailingCommands)) {
        I += Result->NumJobs;
        continue;
      }
    }

    const Command *FailingCommand = nullptr;
    if (int Res = ExecuteCommand(Job, FailingCommand, LogOnly)) {
      FailingCommands.push_back(std::make_pair(Res, FailingCommand));
      // Bail as soon as one command fails in cl driver mode.
      if (TheDriver.IsCLMode())
        return;
    }
    ++I;
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
