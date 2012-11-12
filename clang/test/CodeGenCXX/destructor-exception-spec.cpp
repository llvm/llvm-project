// RUN: %clang_cc1 -emit-llvm-only %s -std=c++11

// PR13479: don't crash with -fno-exceptions.
namespace {
  struct SchedulePostRATDList {
    virtual ~SchedulePostRATDList();
  };

  SchedulePostRATDList::~SchedulePostRATDList() {}

  SchedulePostRATDList Scheduler;
}
