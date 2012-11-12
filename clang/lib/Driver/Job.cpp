//===--- Job.cpp - Command to Execute -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Job.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
using namespace clang::driver;

Job::~Job() {}

void Command::anchor() {}

Command::Command(const Action &_Source, const Tool &_Creator,
                 const char *_Executable, const ArgStringList &_Arguments)
  : Job(CommandClass), Source(_Source), Creator(_Creator),
    Executable(_Executable), Arguments(_Arguments)
{
}

JobList::JobList() : Job(JobListClass) {}

JobList::~JobList() {
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

void JobList::clear() {
  DeleteContainerPointers(Jobs);
}

void Job::addCommand(Command *C) {
  cast<JobList>(this)->addJob(C);
}

