//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_COMPILATION_H_
#define CLANG_DRIVER_COMPILATION_H_

#include "clang/Driver/Job.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace driver {
  class DerivedArgList;
  class Driver;
  class InputArgList;
  class JobList;
  class ToolChain;

/// Compilation - A set of tasks to perform for a single driver
/// invocation.
class Compilation {
  /// The driver we were created by.
  const Driver &TheDriver;

  /// The default tool chain.
  const ToolChain &DefaultToolChain;

  /// The original (untranslated) input argument list.
  InputArgList *Args;

  /// The driver translated arguments. Note that toolchains may perform their
  /// own argument translation.
  DerivedArgList *TranslatedArgs;

  /// The list of actions.
  ActionList Actions;

  /// The root list of jobs.
  JobList Jobs;

  /// Cache of translated arguments for a particular tool chain and bound
  /// architecture.
  llvm::DenseMap<std::pair<const ToolChain*, const char*>,
                 DerivedArgList*> TCArgs;

  /// Temporary files which should be removed on exit.
  ArgStringList TempFiles;

  /// Result files which should be removed on failure.
  ArgStringList ResultFiles;

  /// Result files which are generated correctly on failure, and which should
  /// only be removed if we crash.
  ArgStringList FailureResultFiles;

  /// Redirection for stdout, stderr, etc.
  const llvm::sys::Path **Redirects;

public:
  Compilation(const Driver &D, const ToolChain &DefaultToolChain,
              InputArgList *Args, DerivedArgList *TranslatedArgs);
  ~Compilation();

  const Driver &getDriver() const { return TheDriver; }

  const ToolChain &getDefaultToolChain() const { return DefaultToolChain; }

  const InputArgList &getInputArgs() const { return *Args; }

  const DerivedArgList &getArgs() const { return *TranslatedArgs; }

  DerivedArgList &getArgs() { return *TranslatedArgs; }

  ActionList &getActions() { return Actions; }
  const ActionList &getActions() const { return Actions; }

  JobList &getJobs() { return Jobs; }
  const JobList &getJobs() const { return Jobs; }

  void addCommand(Command *C) { Jobs.addJob(C); }

  const ArgStringList &getTempFiles() const { return TempFiles; }

  const ArgStringList &getResultFiles() const { return ResultFiles; }

  const ArgStringList &getFailureResultFiles() const {
    return FailureResultFiles;
  }

  /// Returns the sysroot path.
  StringRef getSysRoot() const;

  /// getArgsForToolChain - Return the derived argument list for the
  /// tool chain \p TC (or the default tool chain, if TC is not specified).
  ///
  /// \param BoundArch - The bound architecture name, or 0.
  const DerivedArgList &getArgsForToolChain(const ToolChain *TC,
                                            const char *BoundArch);

  /// addTempFile - Add a file to remove on exit, and returns its
  /// argument.
  const char *addTempFile(const char *Name) {
    TempFiles.push_back(Name);
    return Name;
  }

  /// addResultFile - Add a file to remove on failure, and returns its
  /// argument.
  const char *addResultFile(const char *Name) {
    ResultFiles.push_back(Name);
    return Name;
  }

  /// addFailureResultFile - Add a file to remove if we crash, and returns its
  /// argument.
  const char *addFailureResultFile(const char *Name) {
    FailureResultFiles.push_back(Name);
    return Name;
  }

  /// CleanupFileList - Remove the files in the given list.
  ///
  /// \param IssueErrors - Report failures as errors.
  /// \return Whether all files were removed successfully.
  bool CleanupFileList(const ArgStringList &Files,
                       bool IssueErrors=false) const;

  /// PrintJob - Print one job in -### format.
  ///
  /// \param OS - The stream to print on.
  /// \param J - The job to print.
  /// \param Terminator - A string to print at the end of the line.
  /// \param Quote - Should separate arguments be quoted.
  void PrintJob(raw_ostream &OS, const Job &J,
                const char *Terminator, bool Quote) const;

  /// PrintDiagnosticJob - Print one job in -### format, but with the 
  /// superfluous options removed, which are not necessary for 
  /// reproducing the crash.
  ///
  /// \param OS - The stream to print on.
  /// \param J - The job to print.
  void PrintDiagnosticJob(raw_ostream &OS, const Job &J) const;

  /// ExecuteCommand - Execute an actual command.
  ///
  /// \param FailingCommand - For non-zero results, this will be set to the
  /// Command which failed, if any.
  /// \return The result code of the subprocess.
  int ExecuteCommand(const Command &C, const Command *&FailingCommand) const;

  /// ExecuteJob - Execute a single job.
  ///
  /// \param FailingCommand - For non-zero results, this will be set to the
  /// Command which failed.
  /// \return The accumulated result code of the job.
  int ExecuteJob(const Job &J, const Command *&FailingCommand) const;

  /// initCompilationForDiagnostics - Remove stale state and suppress output
  /// so compilation can be reexecuted to generate additional diagnostic
  /// information (e.g., preprocessed source(s)).
  void initCompilationForDiagnostics();
};

} // end namespace driver
} // end namespace clang

#endif
