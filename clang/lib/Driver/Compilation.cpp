//===--- Compilation.cpp - Compilation Task Implementation ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

Compilation::Compilation(const Driver &D, const ToolChain &_DefaultToolChain,
                         InputArgList *_Args, DerivedArgList *_TranslatedArgs)
    : TheDriver(D), DefaultToolChain(_DefaultToolChain),
      CudaHostToolChain(&DefaultToolChain), CudaDeviceToolChain(nullptr),
      Args(_Args), TranslatedArgs(_TranslatedArgs), Redirects(nullptr),
      ForDiagnostics(false) {}

Compilation::~Compilation() {
  delete TranslatedArgs;
  delete Args;

  // Free any derived arg lists.
  for (llvm::DenseMap<std::pair<const ToolChain*, const char*>,
                      DerivedArgList*>::iterator it = TCArgs.begin(),
         ie = TCArgs.end(); it != ie; ++it)
    if (it->second != TranslatedArgs)
      delete it->second;

  // Free the actions, if built.
  for (ActionList::iterator it = Actions.begin(), ie = Actions.end();
       it != ie; ++it)
    delete *it;

  // Free redirections of stdout/stderr.
  if (Redirects) {
    delete Redirects[1];
    delete Redirects[2];
    delete [] Redirects;
  }
}

const DerivedArgList &Compilation::getArgsForToolChain(const ToolChain *TC,
                                                       const char *BoundArch) {
  if (!TC)
    TC = &DefaultToolChain;

  DerivedArgList *&Entry = TCArgs[std::make_pair(TC, BoundArch)];
  if (!Entry) {
    Entry = TC->TranslateArgs(*TranslatedArgs, BoundArch);
    if (!Entry)
      Entry = TranslatedArgs;
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
      getDriver().Diag(clang::diag::err_drv_unable_to_remove_file)
        << EC.message();
    return false;
  }
  return true;
}

bool Compilation::CleanupFileList(const ArgStringList &Files,
                                  bool IssueErrors) const {
  bool Success = true;
  for (ArgStringList::const_iterator
         it = Files.begin(), ie = Files.end(); it != ie; ++it)
    Success &= CleanupFile(*it, IssueErrors);
  return Success;
}

bool Compilation::CleanupFileMap(const ArgStringMap &Files,
                                 const JobAction *JA,
                                 bool IssueErrors) const {
  bool Success = true;
  for (ArgStringMap::const_iterator
         it = Files.begin(), ie = Files.end(); it != ie; ++it) {

    // If specified, only delete the files associated with the JobAction.
    // Otherwise, delete all files in the map.
    if (JA && it->first != JA)
      continue;
    Success &= CleanupFile(it->second, IssueErrors);
  }
  return Success;
}

int Compilation::ExecuteCommand(const Command &C,
                                const Command *&FailingCommand) const {
  if ((getDriver().CCPrintOptions ||
       getArgs().hasArg(options::OPT_v)) && !getDriver().CCGenDiagnostics) {
    raw_ostream *OS = &llvm::errs();

    // Follow gcc implementation of CC_PRINT_OPTIONS; we could also cache the
    // output stream.
    if (getDriver().CCPrintOptions && getDriver().CCPrintOptionsFilename) {
      std::error_code EC;
      OS = new llvm::raw_fd_ostream(getDriver().CCPrintOptionsFilename, EC,
                                    llvm::sys::fs::F_Append |
                                        llvm::sys::fs::F_Text);
      if (EC) {
        getDriver().Diag(clang::diag::err_drv_cc_print_options_failure)
            << EC.message();
        FailingCommand = &C;
        delete OS;
        return 1;
      }
    }

    if (getDriver().CCPrintOptions)
      *OS << "[Logging clang options]";

    C.Print(*OS, "\n", /*Quote=*/getDriver().CCPrintOptions);

    if (OS != &llvm::errs())
      delete OS;
  }

  std::string Error;
  bool ExecutionFailed;
  int Res = C.Execute(Redirects, &Error, &ExecutionFailed);
  if (!Error.empty()) {
    assert(Res && "Error string set with 0 result code!");
    getDriver().Diag(clang::diag::err_drv_command_failure) << Error;
  }

  if (Res)
    FailingCommand = &C;

  return ExecutionFailed ? 1 : Res;
}

typedef SmallVectorImpl< std::pair<int, const Command *> > FailingCommandList;

static bool ActionFailed(const Action *A,
                         const FailingCommandList &FailingCommands) {

  if (FailingCommands.empty())
    return false;

  for (FailingCommandList::const_iterator CI = FailingCommands.begin(),
         CE = FailingCommands.end(); CI != CE; ++CI)
    if (A == &(CI->second->getSource()))
      return true;

  for (Action::const_iterator AI = A->begin(), AE = A->end(); AI != AE; ++AI)
    if (ActionFailed(*AI, FailingCommands))
      return true;

  return false;
}

static bool InputsOk(const Command &C,
                     const FailingCommandList &FailingCommands) {
  return !ActionFailed(&C.getSource(), FailingCommands);
}

void Compilation::ExecuteJobs(const JobList &Jobs,
                              FailingCommandList &FailingCommands) const {
  for (const auto &Job : Jobs) {
    if (!InputsOk(Job, FailingCommands))
      continue;
    const Command *FailingCommand = nullptr;
    if (int Res = ExecuteCommand(Job, FailingCommand))
      FailingCommands.push_back(std::make_pair(Res, FailingCommand));
  }
}

void Compilation::initCompilationForDiagnostics() {
  ForDiagnostics = true;

  // Free actions and jobs.
  DeleteContainerPointers(Actions);
  Jobs.clear();

  // Clear temporary/results file lists.
  TempFiles.clear();
  ResultFiles.clear();
  FailureResultFiles.clear();

  // Remove any user specified output.  Claim any unclaimed arguments, so as
  // to avoid emitting warnings about unused args.
  OptSpecifier OutputOpts[] = { options::OPT_o, options::OPT_MD,
                                options::OPT_MMD };
  for (unsigned i = 0, e = llvm::array_lengthof(OutputOpts); i != e; ++i) {
    if (TranslatedArgs->hasArg(OutputOpts[i]))
      TranslatedArgs->eraseArg(OutputOpts[i]);
  }
  TranslatedArgs->ClaimAllArgs();

  // Redirect stdout/stderr to /dev/null.
  Redirects = new const StringRef*[3]();
  Redirects[0] = nullptr;
  Redirects[1] = new StringRef();
  Redirects[2] = new StringRef();
}

StringRef Compilation::getSysRoot() const {
  return getDriver().SysRoot;
}
