//===-- backtrace.cc --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "gwp_asan/tests/harness.h"

TEST_F(BacktraceGuardedPoolAllocator, DoubleFree) {
  void *Ptr = GPA.allocate(1);
  GPA.deallocate(Ptr);

  std::string DeathRegex = "Double free.*";
  DeathRegex.append("backtrace\\.cpp:25.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("backtrace\\.cpp:15.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("backtrace\\.cpp:14.*");
  ASSERT_DEATH(GPA.deallocate(Ptr), DeathRegex);
}

TEST_F(BacktraceGuardedPoolAllocator, UseAfterFree) {
  char *Ptr = static_cast<char *>(GPA.allocate(1));
  GPA.deallocate(Ptr);

  std::string DeathRegex = "Use after free.*";
  DeathRegex.append("backtrace\\.cpp:40.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("backtrace\\.cpp:30.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("backtrace\\.cpp:29.*");
  ASSERT_DEATH({ *Ptr = 7; }, DeathRegex);
}
