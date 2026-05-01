//===-- examples/Instrumentor/stack_usage.c - An example Instrumentor use -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <inttypes.h>
#include <list>

struct StackTracker {
  std::list<char *> CallStack;
  int64_t FunctionStackUsage = 0;
  int64_t TotalStackUsage = 0;

  std::list<char *> HighWaterMarkCallStack;
  int64_t HighWaterMark = 0;

  ~StackTracker() {
    printf("Stack usage peaked at %" PRId64 " in\n", HighWaterMark);
    HighWaterMarkCallStack.reverse();
    for (char *Name : HighWaterMarkCallStack)
      printf("- %s\n", Name);
  }

  void enter(char *Name) {
    FunctionStackUsage = 0;
    CallStack.push_back(Name);
  }
  void exit(char *Name) {
    CallStack.pop_back();
    TotalStackUsage -= FunctionStackUsage;
  }

  void allocate(int64_t size) {
    TotalStackUsage += size;
    FunctionStackUsage += size;
    if (TotalStackUsage <= HighWaterMark)
      return;
    HighWaterMark = TotalStackUsage;
    HighWaterMarkCallStack = CallStack;
  }
};

static thread_local StackTracker ST;

extern "C" {

void __stack_usage_pre_function(char *Name) { ST.enter(Name); }

void __stack_usage_post_function(char *Name) { ST.exit(Name); }

void __stack_usage_pre_alloca(int64_t size) { ST.allocate(size); }
}
