//===--- Job.h - Commands to Execute ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_JOB_H
#define LLVM_CLANG_DRIVER_JOB_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Option/Option.h"
#include <memory>

namespace llvm {
  class raw_ostream;
}

namespace clang {
namespace driver {
class Action;
class Command;
class Tool;
class InputInfo;

// Re-export this as clang::driver::ArgStringList.
using llvm::opt::ArgStringList;

struct CrashReportInfo {
  StringRef Filename;
  StringRef VFSPath;

  CrashReportInfo(StringRef Filename, StringRef VFSPath)
      : Filename(Filename), VFSPath(VFSPath) {}
};

/// Command - An executable path/name and argument vector to
/// execute.
class Command {
  /// Source - The action which caused the creation of this job.
  const Action &Source;

  /// Tool - The tool which caused the creation of this job.
  const Tool &Creator;

  /// The executable to run.
  const char *Executable;

  /// The list of program arguments (not including the implicit first
  /// argument, which will be the executable).
  llvm::opt::ArgStringList Arguments;

  /// The list of program arguments which are inputs.
  llvm::opt::ArgStringList InputFilenames;

  /// Response file name, if this command is set to use one, or nullptr
  /// otherwise
  const char *ResponseFile;

  /// The input file list in case we need to emit a file list instead of a
  /// proper response file
  llvm::opt::ArgStringList InputFileList;

  /// String storage if we need to create a new argument to specify a response
  /// file
  std::string ResponseFileFlag;

  /// When a response file is needed, we try to put most arguments in an
  /// exclusive file, while others remains as regular command line arguments.
  /// This functions fills a vector with the regular command line arguments,
  /// argv, excluding the ones passed in a response file.
  void buildArgvForResponseFile(llvm::SmallVectorImpl<const char *> &Out) const;

  /// Encodes an array of C strings into a single string separated by whitespace.
  /// This function will also put in quotes arguments that have whitespaces and
  /// will escape the regular backslashes (used in Windows paths) and quotes.
  /// The results are the contents of a response file, written into a raw_ostream.
  void writeResponseFile(raw_ostream &OS) const;

public:
  Command(const Action &Source, const Tool &Creator, const char *Executable,
          const llvm::opt::ArgStringList &Arguments,
          ArrayRef<InputInfo> Inputs);
  // FIXME: This really shouldn't be copyable, but is currently copied in some
  // error handling in Driver::generateCompilationDiagnostics.
  Command(const Command &) = default;
  virtual ~Command() {}

  virtual void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
                     CrashReportInfo *CrashInfo = nullptr) const;

  virtual int Execute(const StringRef **Redirects, std::string *ErrMsg,
                      bool *ExecutionFailed) const;

  /// getSource - Return the Action which caused the creation of this job.
  const Action &getSource() const { return Source; }

  /// getCreator - Return the Tool which caused the creation of this job.
  const Tool &getCreator() const { return Creator; }

  /// Set to pass arguments via a response file when launching the command
  void setResponseFile(const char *FileName);

  /// Set an input file list, necessary if we need to use a response file but
  /// the tool being called only supports input files lists.
  void setInputFileList(llvm::opt::ArgStringList List) {
    InputFileList = std::move(List);
  }

  const char *getExecutable() const { return Executable; }

  const llvm::opt::ArgStringList &getArguments() const { return Arguments; }

  /// Print a command argument, and optionally quote it.
  static void printArg(llvm::raw_ostream &OS, const char *Arg, bool Quote);
};

/// Like Command, but with a fallback which is executed in case
/// the primary command crashes.
class FallbackCommand : public Command {
public:
  FallbackCommand(const Action &Source_, const Tool &Creator_,
                  const char *Executable_, const ArgStringList &Arguments_,
                  ArrayRef<InputInfo> Inputs,
                  std::unique_ptr<Command> Fallback_);

  void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
             CrashReportInfo *CrashInfo = nullptr) const override;

  int Execute(const StringRef **Redirects, std::string *ErrMsg,
              bool *ExecutionFailed) const override;

private:
  std::unique_ptr<Command> Fallback;
};

/// Like Command, but always pretends that the wrapped command succeeded.
class ForceSuccessCommand : public Command {
public:
  ForceSuccessCommand(const Action &Source_, const Tool &Creator_,
                      const char *Executable_, const ArgStringList &Arguments_,
                      ArrayRef<InputInfo> Inputs);

  void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
             CrashReportInfo *CrashInfo = nullptr) const override;

  int Execute(const StringRef **Redirects, std::string *ErrMsg,
              bool *ExecutionFailed) const override;
};

/// JobList - A sequence of jobs to perform.
class JobList {
public:
  typedef SmallVector<std::unique_ptr<Command>, 4> list_type;
  typedef list_type::size_type size_type;
  typedef llvm::pointee_iterator<list_type::iterator> iterator;
  typedef llvm::pointee_iterator<list_type::const_iterator> const_iterator;

private:
  list_type Jobs;

public:
  void Print(llvm::raw_ostream &OS, const char *Terminator,
             bool Quote, CrashReportInfo *CrashInfo = nullptr) const;

  /// Add a job to the list (taking ownership).
  void addJob(std::unique_ptr<Command> J) { Jobs.push_back(std::move(J)); }

  /// Clear the job list.
  void clear();

  const list_type &getJobs() const { return Jobs; }

  size_type size() const { return Jobs.size(); }
  iterator begin() { return Jobs.begin(); }
  const_iterator begin() const { return Jobs.begin(); }
  iterator end() { return Jobs.end(); }
  const_iterator end() const { return Jobs.end(); }
};

} // end namespace driver
} // end namespace clang

#endif
