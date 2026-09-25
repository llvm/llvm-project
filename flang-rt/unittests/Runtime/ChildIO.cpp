//===-- unittests/Runtime/ChildIO.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include "flang-rt/runtime/unit.h"
#include "flang/Runtime/io-api.h"

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

struct ChildIOTests : CrashHandlerFixture {};

TEST(ChildIOTests, InNamelist) {
  // CHARACTER(LEN=10) :: output
  static constexpr int bufferSize{10};
  char output[bufferSize];

  // WRITE(UNIT=output,NML=...)
  Cookie parentCookie{IONAME(BeginInternalListOutput)(output, bufferSize)};
  parentCookie->mutableModes().inNamelist = true;

  // invoke child I/O (DTIO)
  IoErrorHandler &handler{parentCookie->GetIoErrorHandler()};
  ExternalFileUnit *newUnit{&ExternalFileUnit::NewUnit(handler, true)};
  [[maybe_unused]] ChildIo &child{newUnit->PushChildIo(*parentCookie)};
  Cookie childCookie{IONAME(BeginExternalListOutput)(
      newUnit->unitNumber(), __FILE__, __LINE__)};
  ASSERT_FALSE(childCookie->mutableModes().inNamelist);
}
