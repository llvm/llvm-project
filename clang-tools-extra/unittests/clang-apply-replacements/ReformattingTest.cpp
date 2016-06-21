//===- clang-apply-replacements/ReformattingTest.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "common/VirtualFileHelper.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Refactoring.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::replace;

typedef std::vector<clang::tooling::Replacement> ReplacementsVec;

static Replacement makeReplacement(unsigned Offset, unsigned Length,
                                   unsigned ReplacementLength,
                                   llvm::StringRef FilePath = "") {
  return Replacement(FilePath, Offset, Length,
                     std::string(ReplacementLength, '~'));
}

// generate a set of replacements containing one element
static ReplacementsVec makeReplacements(unsigned Offset, unsigned Length,
                                        unsigned ReplacementLength,
                                        llvm::StringRef FilePath = "~") {
  ReplacementsVec Replaces;
  Replaces.push_back(
      makeReplacement(Offset, Length, ReplacementLength, FilePath));
  return Replaces;
}

// Put these functions in the clang::tooling namespace so arg-dependent name
// lookup finds these functions for the EXPECT_EQ macros below.
namespace clang {
namespace tooling {

std::ostream &operator<<(std::ostream &os, const Range &R) {
  return os << "Range(" << R.getOffset() << ", " << R.getLength() << ")";
}

} // namespace tooling
} // namespace clang

// Ensure zero-length ranges are produced. Even lines where things are deleted
// need reformatting.
TEST(CalculateChangedRangesTest, producesZeroLengthRange) {
  RangeVector Changes = calculateChangedRanges(makeReplacements(0, 4, 0));
  EXPECT_EQ(Range(0, 0), Changes.front());
}

// Basic test to ensure replacements turn into ranges properly.
TEST(CalculateChangedRangesTest, calculatesRanges) {
  ReplacementsVec R;
  R.push_back(makeReplacement(2, 0, 3));
  R.push_back(makeReplacement(5, 2, 4));
  RangeVector Changes = calculateChangedRanges(R);

  Range ExpectedRanges[] = { Range(2, 3), Range(8, 4) };
  EXPECT_TRUE(std::equal(Changes.begin(), Changes.end(), ExpectedRanges));
}
