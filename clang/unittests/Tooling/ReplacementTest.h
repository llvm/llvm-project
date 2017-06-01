//===- unittest/Tooling/ReplacementTest.h - Replacements related test------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
static tooling::Replacements
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
