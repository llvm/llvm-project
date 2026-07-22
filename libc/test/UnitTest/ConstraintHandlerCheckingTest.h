//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a test fixture for checking Annex K's constraint handler
/// behavior.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H
#define LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H

#include "hdr/types/constraint_handler_t.h"
#include "hdr/types/errno_t.h"
#include "src/stdlib/set_constraint_handler_s.h"
#include "test/UnitTest/Test.h"

namespace {

bool error_flag;

// This function conforms to the constraint_handler_t signature, which is the
// argument type for set_constraint_handler_s used below. None of the constraint
// handler's arguments are used in this test fixture.
void local_constraint_handler(const char *__restrict /*msg*/,
                              void *__restrict /*ptr*/, errno_t /*error*/) {
  error_flag = true;
}

} // anonymous namespace

namespace LIBC_NAMESPACE_DECL {

namespace testing {

class ConstraintHandlerCheckingTest : public Test {
public:
  void SetUp() override {
    Test::SetUp();
    error_flag = false;
    LIBC_NAMESPACE::set_constraint_handler_s(local_constraint_handler);
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_UNITTEST_CONSTRAINTHANDLERCHECKINGTEST_H
