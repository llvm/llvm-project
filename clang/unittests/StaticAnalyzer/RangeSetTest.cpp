//===- unittests/StaticAnalyzer/RangeSetTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

// TestCase contains to lists of ranges.
// Original one has to be negated.
// Expected one has to be compared to negated original range.
template <typename T> struct TestCase {
  RangeSet original;
  RangeSet expected;

  TestCase(BasicValueFactory &BVF, RangeSet::Factory &F,
           const std::initializer_list<T> &originalList,
           const std::initializer_list<T> &expectedList)
      : original(createRangeSetFromList(BVF, F, originalList)),
        expected(createRangeSetFromList(BVF, F, expectedList)) {}

private:
  RangeSet createRangeSetFromList(BasicValueFactory &BVF, RangeSet::Factory &F,
                                  const std::initializer_list<T> rangeList) {
    llvm::APSInt from(sizeof(T) * 8, std::is_unsigned<T>::value);
    llvm::APSInt to = from;
    RangeSet rangeSet = F.getEmptySet();
    for (auto it = rangeList.begin(); it != rangeList.end(); it += 2) {
      from = *it;
      to = *(it + 1);
      rangeSet = rangeSet.addRange(
          F, RangeSet(F, BVF.getValue(from), BVF.getValue(to)));
    }
    return rangeSet;
  }

  void printNegate(const TestCase &TestCase) {
    TestCase.original.print(llvm::dbgs());
    llvm::dbgs() << " => ";
    TestCase.expected.print(llvm::dbgs());
  }
};

class RangeSetTest : public testing::Test {
protected:
  // Init block
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCode("struct foo;");
  ASTContext &context = AST->getASTContext();
  llvm::BumpPtrAllocator alloc;
  BasicValueFactory BVF{context, alloc};
  RangeSet::Factory F;
  // End init block

  template <typename T> void checkNegate() {
    using type = T;

    // Use next values of the range {MIN, A, B, MID, C, D, MAX}.

    // MID is a value in the middle of the range
    // which unary minus does not affect on,
    // e.g. int8/int32(0), uint8(128), uint32(2147483648).

    constexpr type MIN = std::numeric_limits<type>::min();
    constexpr type MAX = std::numeric_limits<type>::max();
    constexpr type MID = std::is_signed<type>::value
                             ? 0
                             : ~(static_cast<type>(-1) / static_cast<type>(2));
    constexpr type A = MID - static_cast<type>(42 + 42);
    constexpr type B = MID - static_cast<type>(42);
    constexpr type C = -B;
    constexpr type D = -A;

    static_assert(MIN < A && A < B && B < MID && MID < C && C < D && D < MAX,
                  "Values shall be in an ascending order");

    // Left {[x, y], [x, y]} is what shall be negated.
    // Right {[x, y], [x, y]} is what shall be compared to a negation result.
    TestCase<type> cases[] = {
        {BVF, F, {MIN, A}, {MIN, MIN, D, MAX}},
        {BVF, F, {MIN, C}, {MIN, MIN, B, MAX}},
        {BVF, F, {MIN, MID}, {MIN, MIN, MID, MAX}},
        {BVF, F, {MIN, MAX}, {MIN, MAX}},
        {BVF, F, {A, D}, {A, D}},
        {BVF, F, {A, B}, {C, D}},
        {BVF, F, {MIN, A, D, MAX}, {MIN, A, D, MAX}},
        {BVF, F, {MIN, B, MID, D}, {MIN, MIN, A, MID, C, MAX}},
        {BVF, F, {MIN, MID, C, D}, {MIN, MIN, A, B, MID, MAX}},
        {BVF, F, {MIN, MID, C, MAX}, {MIN, B, MID, MAX}},
        {BVF, F, {A, MID, D, MAX}, {MIN + 1, A, MID, D}},
        {BVF, F, {A, A}, {D, D}},
        {BVF, F, {MID, MID}, {MID, MID}},
        {BVF, F, {MAX, MAX}, {MIN + 1, MIN + 1}},
    };

    for (const auto &c : cases) {
      // Negate original and check with expected.
      RangeSet negatedFromOriginal = c.original.Negate(BVF, F);
      EXPECT_EQ(negatedFromOriginal, c.expected);
      // Negate negated back and check with original.
      RangeSet negatedBackward = negatedFromOriginal.Negate(BVF, F);
      EXPECT_EQ(negatedBackward, c.original);
    }
  }
};

TEST_F(RangeSetTest, RangeSetNegateTest) {
  checkNegate<int8_t>();
  checkNegate<uint8_t>();
  checkNegate<int16_t>();
  checkNegate<uint16_t>();
  checkNegate<int32_t>();
  checkNegate<uint32_t>();
  checkNegate<int64_t>();
  checkNegate<uint64_t>();
}

} // namespace
} // namespace ento
} // namespace clang
