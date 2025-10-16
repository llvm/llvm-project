//===- UncheckedStatusOrAccessModelTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "UncheckedStatusOrAccessModelTestFixture.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.h"
#include "gtest/gtest.h"

namespace clang::dataflow::statusor_model {
namespace {

INSTANTIATE_TEST_SUITE_P(
    UncheckedStatusOrAccessModelTest, UncheckedStatusOrAccessModelTest,
    testing::Values(
        std::make_pair(new UncheckedStatusOrAccessModelTestExecutor<
                           UncheckedStatusOrAccessModel>(),
                       UncheckedStatusOrAccessModelTestAliasKind::kUnaliased),
        std::make_pair(
            new UncheckedStatusOrAccessModelTestExecutor<
                UncheckedStatusOrAccessModel>(),
            UncheckedStatusOrAccessModelTestAliasKind::kPartiallyAliased),
        std::make_pair(
            new UncheckedStatusOrAccessModelTestExecutor<
                UncheckedStatusOrAccessModel>(),
            UncheckedStatusOrAccessModelTestAliasKind::kFullyAliased)));
} // namespace

} // namespace clang::dataflow::statusor_model
