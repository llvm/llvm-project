//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_COMPILATION_H
#define LLVM_CLANG_DRIVER_COMPILATION_H

#include "clang/Driver/Job.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace opt {
  class DerivedArgList;
  class InputArgList;
}
}

namespace clang {
namespace driver {
  class Driver;
  class JobAction;
  class JobList;
  class ToolChain;

/// Compilation - A set of tasks to perform for a single driver
/// invocation.
class Compilation {
  /// The driver we were created by.
  const Driver &TheDriver;

  /// The default tool chain.
  const ToolChain &DefaultToolChain;

  const ToolChain *CudaHostToolChain;
  const ToolChain *CudaDeviceToolChain;

  /// The original (untranslated) input argument list.
  llvm::opt::InputArgList *Args;

  /// The driver translated arguments. Note that toolchains may perform their
  /// own argument translation.
  llvm::opt::DerivedArgList *TranslatedArgs;

  /// The list of actions.
  ActionList Actions;

  /// The root list of jobs.
  JobList Jobs;

  /// Cache of translated arguments for a particular tool chain and bound
  /// architecture.
  llvm::DenseMap<std::pair<const ToolChain *, const char *>,
                 llvm::opt::DerivedArgList *> TCArgs;

  /// Temporary files which should be removed on exit.
  llvm::opt::ArgStringList TempFiles;

  /// Result files which should be removed on failure.
  ArgStringMap ResultFiles;

  /// Result files which are generated correctly on failure, and which should
  /// only be removed if we crash.
  ArgStringMap FailureResultFiles;

  /// Redirection for stdout, stderr, etc.
  const StringRef **Redirects;

  /// Whether we're compiling for diagnostic purposes.
  bool ForDiagnostics;

public:
  Compilation(const Driver &D, const ToolChain &DefaultToolChain,
              llvm::opt::InputArgList *Args,
              llvm::opt::DerivedArgList *TranslatedArgs);
  ~Compilation();

  const Driver &getDriver() const { return TheDriver; }

  const ToolChain &getDefaultToolChain() const { return DefaultToolChain; }
  const ToolChain *getCudaHostToolChain() const { return CudaHostToolChain; }
  const ToolChain *getCudaDeviceToolChain() const {
    return CudaDeviceToolChain;
  }

  void setCudaHostToolChain(const ToolChain *HostToolChain) {
    CudaHostToolChain = HostToolChain;
  }
  void setCudaDeviceToolChain(const ToolChain *DeviceToolChain) {
    CudaDeviceToolChain = DeviceToolChain;
  }

  const llvm::opt::InputArgList &getInputArgs() const { return *Args; }

  const llvm::opt::DerivedArgList &getArgs() const { return *TranslatedArgs; }

  llvm::opt::DerivedArgList &getArgs() { return *TranslatedArgs; }

  ActionList &getActions() { return Actions; }
  const ActionList &getActions() const { return Actions; }

  JobList &getJobs() { return Jobs; }
  const JobList &getJobs() const { return Jobs; }

  void addCommand(std::unique_ptr<Command> C) { Jobs.addJob(std::move(C)); }

  const llvm::opt::ArgStringList &getTempFiles() const { return TempFiles; }

  const ArgStringMap &getResultFiles() const { return ResultFiles; }

  const ArgStringMap &getFailureResultFiles() const {
    return FailureResultFiles;
  }

  /// Returns the sysroot path.
  StringRef getSysRoot() const;

  /// getArgsForToolChain - Return the derived argument list for the
  /// tool chain \p TC (or the default tool chain, if TC is not specified).
  ///
  /// \param BoundArch - The bound architecture name, or 0.
  const llvm::opt::DerivedArgList &getArgsForToolChain(const ToolChain *TC,
                                                       const char *BoundArch);

  /// addTempFile - Add a file to remove on exit, and returns its
  /// argument.
  const char *addTempFile(const char *Name) {
    TempFiles.push_back(Name);
    return Name;
  }

  /// addResultFile - Add a file to remove on failure, and returns its
  /// argument.
  const char *addResultFile(const char *Name, const JobAction *JA) {
    ResultFiles[JA] = Name;
    return Name;
  }

  /// addFailureResultFile - Add a file to remove if we crash, and returns its
  /// argument.
  const char *addFailureResultFile(const char *Name, const JobAction *JA) {
    FailureResultFiles[JA] = Name;
    return Name;
  }

  /// CleanupFile - Delete a given file.
  ///
  /// \param IssueErrors - Report failures as errors.
  /// \return Whether the file was removed successfully.
  bool CleanupFile(const char *File, bool IssueErrors = false) const;

  /// CleanupFileList - Remove the files in the given list.
  ///
  /// \param IssueErrors - Report failures as errors.
  /// \return Whether all files were removed successfully.
  bool CleanupFileList(const llvm::opt::ArgStringList &Files,
                       bool IssueErrors = false) const;

  /// CleanupFileMap - Remove the files in the given map.
  ///
  /// \param JA - If specified, only delete the files associated with this
  /// JobAction.  Otherwise, delete all files in the map.
  /// \param IssueErrors - Report failures as errors.
  /// \return Whether all files were removed successfully.
  bool CleanupFileMap(const ArgStringMap &Files,
                      const JobAction *JA,
                      bool IssueErrors = false) const;

  /// ExecuteCommand - Execute an actual command.
  ///
  /// \param FailingCommand - For non-zero results, this will be set to the
  /// Command which failed, if any.
  /// \return The result code of the subprocess.
  int ExecuteCommand(const Command &C, const Command *&FailingCommand) const;

  /// ExecuteJob - Execute a single job.
  ///
  /// \param FailingCommands - For non-zero results, this will be a vector of
  /// failing commands and their associated result code.
  void ExecuteJobs(
      const JobList &Jobs,
      SmallVectorImpl<std::pair<int, const Command *>> &FailingCommands) const;

  /// initCompilationForDiagnostics - Remove stale state and suppress output
  /// so compilation can be reexecuted to generate additional diagnostic
  /// information (e.g., preprocessed source(s)).
  void initCompilationForDiagnostics();

  /// Return true if we're compiling for diagnostics.
  bool isForDiagnostics() { return ForDiagnostics; }
};

} // end namespace driver
} // end namespace clang

#endif
