//===- unittest/Tooling/ReplacementTest.h - Replacements related test------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines utility class and function for Replacement related tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_TOOLING_REPLACEMENTTESTBASE_H
#define LLVM_CLANG_UNITTESTS_TOOLING_REPLACEMENTTESTBASE_H

#include "RewriterTestContext.h"
#include "clang/Tooling/Core/Replacement.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

/// \brief Converts a set of replacements to Replacements class.
/// \return A Replacements class containing \p Replaces on success; otherwise,
/// an empty Replacements is returned.
inline tooling::Replacements
toReplacements(const std::set<tooling::Replacement> &Replaces) {
  tooling::Replacements Result;
  for (const auto &R : Replaces) {
    auto Err = Result.add(R);
    EXPECT_TRUE(!Err);
    if (Err) {
      llvm::errs() << llvm::toString(std::move(Err)) << "\n";
      return tooling::Replacements();
    }
  }
  return Result;
}

/// \brief A utility class for replacement related tests.
class ReplacementTest : public ::testing::Test {
protected:
  tooling::Replacement createReplacement(SourceLocation Start, unsigned Length,
                                         llvm::StringRef ReplacementText) {
    return tooling::Replacement(Context.Sources, Start, Length,
                                ReplacementText);
  }

  RewriterTestContext Context;
};

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_TOOLING_REPLACEMENTTESTBASE_H
