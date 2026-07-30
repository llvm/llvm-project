//===- unittests/Analysis/FlowSensitive/WatchedLiteralsSolverTest.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "SolverTest.h"

namespace clang::dataflow::test {

template <>
WatchedLiteralsSolver
SolverTest<WatchedLiteralsSolver>::createSolverWithLowTimeout() {
  return WatchedLiteralsSolver(10);
}

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(WatchedLiteralsSolverTest, SolverTest,
                               WatchedLiteralsSolver, );

} // namespace
} // namespace clang::dataflow::test
