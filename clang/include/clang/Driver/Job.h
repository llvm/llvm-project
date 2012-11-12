//===--- Job.h - Commands to Execute ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_JOB_H_
#define CLANG_DRIVER_JOB_H_

#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/LLVM.h"

namespace clang {
namespace driver {
class Command;
class Tool;

class Job {
public:
  enum JobClass {
    CommandClass,
    JobListClass
  };

private:
  JobClass Kind;

protected:
  Job(JobClass _Kind) : Kind(_Kind) {}
public:
  virtual ~Job();

  JobClass getKind() const { return Kind; }

  /// addCommand - Append a command to the current job, which must be
  /// either a piped job or a job list.
  void addCommand(Command *C);
};

  /// Command - An executable path/name and argument vector to
  /// execute.
class Command : public Job {
  virtual void anchor();

  /// Source - The action which caused the creation of this job.
  const Action &Source;

  /// Tool - The tool which caused the creation of this job.
  const Tool &Creator;

  /// The executable to run.
  const char *Executable;

  /// The list of program arguments (not including the implicit first
  /// argument, which will be the executable).
  ArgStringList Arguments;

public:
  Command(const Action &_Source, const Tool &_Creator, const char *_Executable,
          const ArgStringList &_Arguments);

  /// getSource - Return the Action which caused the creation of this job.
  const Action &getSource() const { return Source; }

  /// getCreator - Return the Tool which caused the creation of this job.
  const Tool &getCreator() const { return Creator; }

  const char *getExecutable() const { return Executable; }

  const ArgStringList &getArguments() const { return Arguments; }

  static bool classof(const Job *J) {
    return J->getKind() == CommandClass;
  }
};

  /// JobList - A sequence of jobs to perform.
class JobList : public Job {
public:
  typedef SmallVector<Job*, 4> list_type;
  typedef list_type::size_type size_type;
  typedef list_type::iterator iterator;
  typedef list_type::const_iterator const_iterator;

private:
  list_type Jobs;

public:
  JobList();
  virtual ~JobList();

  /// Add a job to the list (taking ownership).
  void addJob(Job *J) { Jobs.push_back(J); }

  /// Clear the job list.
  void clear();

  const list_type &getJobs() const { return Jobs; }

  size_type size() const { return Jobs.size(); }
  iterator begin() { return Jobs.begin(); }
  const_iterator begin() const { return Jobs.begin(); }
  iterator end() { return Jobs.end(); }
  const_iterator end() const { return Jobs.end(); }

  static bool classof(const Job *J) {
    return J->getKind() == JobListClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
