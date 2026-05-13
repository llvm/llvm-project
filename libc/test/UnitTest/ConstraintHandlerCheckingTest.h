//===-- ConstraintHandlerCheckingTest.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H
#define LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H

#include "hdr/types/constraint_handler_t.h"
#include "hdr/types/errno_t.h"
#include "src/stdlib/set_constraint_handler_s.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/Test.h"

namespace {

char buffer[300];

void local_constraint_handler(const char *__restrict msg,
                              void *__restrict /*ptr*/, errno_t /*error*/) {
  LIBC_NAMESPACE::internal::strlcpy(buffer, msg, sizeof(buffer));
}

} // anonymous namespace

namespace LIBC_NAMESPACE_DECL {

namespace testing {

class ConstraintHandlerCheckingTest : public Test {
public:
  void SetUp() override {
    Test::SetUp();
    LIBC_NAMESPACE::set_constraint_handler_s(local_constraint_handler);
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H
