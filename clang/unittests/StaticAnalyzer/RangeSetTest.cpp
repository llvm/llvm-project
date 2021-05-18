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
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

namespace clang {
namespace ento {

template <class RangeOrSet> static std::string toString(const RangeOrSet &Obj) {
  std::string ObjRepresentation;
  llvm::raw_string_ostream SS(ObjRepresentation);
  Obj.dump(SS);
  return SS.str();
}
LLVM_ATTRIBUTE_UNUSED static std::string toString(const llvm::APSInt &Point) {
  return Point.toString(10);
}
// We need it here for better fail diagnostics from gtest.
LLVM_ATTRIBUTE_UNUSED static std::ostream &operator<<(std::ostream &OS,
                                                      const RangeSet &Set) {
  return OS << toString(Set);
}

} // namespace ento
} // namespace clang

namespace {

template <typename BaseType> class RangeSetTest : public testing::Test {
public:
  // Init block
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCode("struct foo;");
  ASTContext &Context = AST->getASTContext();
  llvm::BumpPtrAllocator Arena;
  BasicValueFactory BVF{Context, Arena};
  RangeSet::Factory F{BVF};
  // End init block

  using Self = RangeSetTest<BaseType>;
  using RawRange = std::pair<BaseType, BaseType>;
  using RawRangeSet = std::initializer_list<RawRange>;

  static constexpr BaseType getMin() {
    return std::numeric_limits<BaseType>::min();
  }
  static constexpr BaseType getMax() {
    return std::numeric_limits<BaseType>::max();
  }
  static constexpr BaseType getMid() {
    return isSigned() ? 0 : ~(fromInt(-1) / fromInt(2));
  }

  static constexpr bool isSigned() { return std::is_signed<BaseType>::value; }
  static constexpr BaseType fromInt(int X) { return static_cast<BaseType>(X); }

  static llvm::APSInt Base;
  const llvm::APSInt &from(BaseType X) {
    llvm::APSInt Dummy = Base;
    Dummy = X;
    return BVF.getValue(Dummy);
  }

  Range from(const RawRange &Init) {
    return Range(from(Init.first), from(Init.second));
  }

  RangeSet from(const RawRangeSet &Init) {
    RangeSet RangeSet = F.getEmptySet();
    for (const auto &Raw : Init) {
      RangeSet = F.add(RangeSet, from(Raw));
    }
    return RangeSet;
  }

  template <class F, class... RawArgTypes>
  void wrap(F ActualFunction, RawArgTypes &&... Args) {
    (this->*ActualFunction)(from(std::forward<RawArgTypes>(Args))...);
  }

  void checkNegateImpl(RangeSet Original, RangeSet Expected) {
    RangeSet NegatedFromOriginal = F.negate(Original);
    EXPECT_EQ(NegatedFromOriginal, Expected);
    // Negate negated back and check with original.
    RangeSet NegatedBackward = F.negate(NegatedFromOriginal);
    EXPECT_EQ(NegatedBackward, Original);
  }

  void checkNegate(RawRangeSet RawOriginal, RawRangeSet RawExpected) {
    wrap(&Self::checkNegateImpl, RawOriginal, RawExpected);
  }

  template <class PointOrSet>
  void checkIntersectImpl(RangeSet LHS, PointOrSet RHS, RangeSet Expected) {
    RangeSet Result = F.intersect(LHS, RHS);
    EXPECT_EQ(Result, Expected)
        << "while intersecting " << toString(LHS) << " and " << toString(RHS);
  }

  void checkIntersectRangeImpl(RangeSet LHS, const llvm::APSInt &Lower,
                               const llvm::APSInt &Upper, RangeSet Expected) {
    RangeSet Result = F.intersect(LHS, Lower, Upper);
    EXPECT_EQ(Result, Expected)
        << "while intersecting " << toString(LHS) << " and [" << toString(Lower)
        << ", " << toString(Upper) << "]";
  }

  void checkIntersect(RawRangeSet RawLHS, RawRangeSet RawRHS,
                      RawRangeSet RawExpected) {
    wrap(&Self::checkIntersectImpl<RangeSet>, RawLHS, RawRHS, RawExpected);
  }

  void checkIntersect(RawRangeSet RawLHS, BaseType RawRHS,
                      RawRangeSet RawExpected) {
    wrap(&Self::checkIntersectImpl<const llvm::APSInt &>, RawLHS, RawRHS,
         RawExpected);
  }

  void checkIntersect(RawRangeSet RawLHS, BaseType RawLower, BaseType RawUpper,
                      RawRangeSet RawExpected) {
    wrap(&Self::checkIntersectRangeImpl, RawLHS, RawLower, RawUpper,
         RawExpected);
  }

  void checkContainsImpl(RangeSet LHS, const llvm::APSInt &RHS, bool Expected) {
    bool Result = LHS.contains(RHS);
    EXPECT_EQ(Result, Expected)
        << toString(LHS) << (Result ? " contains " : " doesn't contain ")
        << toString(RHS);
  }

  void checkContains(RawRangeSet RawLHS, BaseType RawRHS, bool Expected) {
    checkContainsImpl(from(RawLHS), from(RawRHS), Expected);
  }

  template <class RHSType>
  void checkAddImpl(RangeSet LHS, RHSType RHS, RangeSet Expected) {
    RangeSet Result = F.add(LHS, RHS);
    EXPECT_EQ(Result, Expected)
        << "while adding " << toString(LHS) << " and " << toString(RHS);
  }

  void checkAdd(RawRangeSet RawLHS, RawRange RawRHS, RawRangeSet RawExpected) {
    wrap(&Self::checkAddImpl<Range>, RawLHS, RawRHS, RawExpected);
  }

  void checkAdd(RawRangeSet RawLHS, RawRangeSet RawRHS,
                RawRangeSet RawExpected) {
    wrap(&Self::checkAddImpl<RangeSet>, RawRHS, RawLHS, RawExpected);
  }

  void checkAdd(RawRangeSet RawLHS, BaseType RawRHS, RawRangeSet RawExpected) {
    wrap(&Self::checkAddImpl<const llvm::APSInt &>, RawLHS, RawRHS,
         RawExpected);
  }

  void checkDeleteImpl(const llvm::APSInt &Point, RangeSet From,
                       RangeSet Expected) {
    RangeSet Result = F.deletePoint(From, Point);
    EXPECT_EQ(Result, Expected)
        << "while deleting " << toString(Point) << " from " << toString(From);
  }

  void checkDelete(BaseType Point, RawRangeSet RawFrom,
                   RawRangeSet RawExpected) {
    wrap(&Self::checkDeleteImpl, Point, RawFrom, RawExpected);
  }
};

} // namespace

template <typename BaseType>
llvm::APSInt RangeSetTest<BaseType>::Base{sizeof(BaseType) * 8, !isSigned()};

using IntTypes = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t,
                                  uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(RangeSetTest, IntTypes, );

TYPED_TEST(RangeSetTest, RangeSetNegateTest) {
  // Use next values of the range {MIN, A, B, MID, C, D, MAX}.

  constexpr TypeParam MIN = TestFixture::getMin();
  constexpr TypeParam MAX = TestFixture::getMax();
  // MID is a value in the middle of the range
  // which unary minus does not affect on,
  // e.g. int8/int32(0), uint8(128), uint32(2147483648).
  constexpr TypeParam MID = TestFixture::getMid();
  constexpr TypeParam A = MID - TestFixture::fromInt(42 + 42);
  constexpr TypeParam B = MID - TestFixture::fromInt(42);
  constexpr TypeParam C = -B;
  constexpr TypeParam D = -A;

  static_assert(MIN < A && A < B && B < MID && MID < C && C < D && D < MAX,
                "Values shall be in an ascending order");

  this->checkNegate({{MIN, A}}, {{MIN, MIN}, {D, MAX}});
  this->checkNegate({{MIN, C}}, {{MIN, MIN}, {B, MAX}});
  this->checkNegate({{MIN, MID}}, {{MIN, MIN}, {MID, MAX}});
  this->checkNegate({{MIN, MAX}}, {{MIN, MAX}});
  this->checkNegate({{A, D}}, {{A, D}});
  this->checkNegate({{A, B}}, {{C, D}});
  this->checkNegate({{MIN, A}, {D, MAX}}, {{MIN, A}, {D, MAX}});
  this->checkNegate({{MIN, B}, {MID, D}}, {{MIN, MIN}, {A, MID}, {C, MAX}});
  this->checkNegate({{MIN, MID}, {C, D}}, {{MIN, MIN}, {A, B}, {MID, MAX}});
  this->checkNegate({{MIN, MID}, {C, MAX}}, {{MIN, B}, {MID, MAX}});
  this->checkNegate({{A, MID}, {D, MAX}}, {{MIN + 1, A}, {MID, D}});
  this->checkNegate({{A, A}}, {{D, D}});
  this->checkNegate({{MID, MID}}, {{MID, MID}});
  this->checkNegate({{MAX, MAX}}, {{MIN + 1, MIN + 1}});
}

TYPED_TEST(RangeSetTest, RangeSetPointIntersectTest) {
  // Check that we can correctly intersect empty sets.
  this->checkIntersect({}, 42, {});
  // Check that intersection with itself produces the same set.
  this->checkIntersect({{42, 42}}, 42, {{42, 42}});
  // Check more general cases.
  this->checkIntersect({{0, 10}, {20, 30}, {30, 40}, {50, 60}}, 42, {});
  this->checkIntersect({{0, 10}, {20, 30}, {30, 60}}, 42, {{42, 42}});
}

TYPED_TEST(RangeSetTest, RangeSetRangeIntersectTest) {
  constexpr TypeParam MIN = TestFixture::getMin();
  constexpr TypeParam MAX = TestFixture::getMax();

  // Check that we can correctly intersect empty sets.
  this->checkIntersect({}, 10, 20, {});
  this->checkIntersect({}, 20, 10, {});
  // Check that intersection with itself produces the same set.
  this->checkIntersect({{10, 20}}, 10, 20, {{10, 20}});
  this->checkIntersect({{MIN, 10}, {20, MAX}}, 20, 10, {{MIN, 10}, {20, MAX}});
  // Check non-overlapping range intersections.
  this->checkIntersect({{10, 20}}, 21, 9, {});
  this->checkIntersect({{MIN, 9}, {21, MAX}}, 10, 20, {});
  // Check more general cases.
  this->checkIntersect({{0, 10}, {20, 30}, {30, 40}, {50, 60}}, 10, 35,
                       {{10, 10}, {20, 30}, {30, 35}});
  this->checkIntersect({{0, 10}, {20, 30}, {30, 40}, {50, 60}}, 35, 10,
                       {{0, 10}, {35, 40}, {50, 60}});
}

TYPED_TEST(RangeSetTest, RangeSetGenericIntersectTest) {
  // Check that we can correctly intersect empty sets.
  this->checkIntersect({}, {}, {});
  this->checkIntersect({}, {{0, 10}}, {});
  this->checkIntersect({{0, 10}}, {}, {});

  this->checkIntersect({{0, 10}}, {{4, 6}}, {{4, 6}});
  this->checkIntersect({{0, 10}}, {{4, 20}}, {{4, 10}});
  // Check that intersection with points works as expected.
  this->checkIntersect({{0, 10}}, {{4, 4}}, {{4, 4}});
  // All ranges are closed, check that intersection with edge points works as
  // expected.
  this->checkIntersect({{0, 10}}, {{10, 10}}, {{10, 10}});

  // Let's check that we can skip some intervals and partially intersect
  // other intervals.
  this->checkIntersect({{0, 2}, {4, 5}, {6, 9}, {10, 11}, {12, 12}, {13, 15}},
                       {{8, 14}, {20, 30}},
                       {{8, 9}, {10, 11}, {12, 12}, {13, 14}});
  // Check more generic case.
  this->checkIntersect(
      {{0, 1}, {2, 3}, {5, 6}, {7, 15}, {25, 30}},
      {{4, 10}, {11, 11}, {12, 16}, {17, 17}, {19, 20}, {21, 23}, {24, 27}},
      {{5, 6}, {7, 10}, {11, 11}, {12, 15}, {25, 27}});
}

TYPED_TEST(RangeSetTest, RangeSetContainsTest) {
  // Check with an empty set.
  this->checkContains({}, 10, false);
  // Check contains with sets of size one:
  //   * when the whole range is less
  this->checkContains({{0, 5}}, 10, false);
  //   * when the whole range is greater
  this->checkContains({{20, 25}}, 10, false);
  //   * when the range is just the point we are looking for
  this->checkContains({{10, 10}}, 10, true);
  //   * when the range starts with the point
  this->checkContains({{10, 15}}, 10, true);
  //   * when the range ends with the point
  this->checkContains({{5, 10}}, 10, true);
  //   * when the range has the point somewhere in the middle
  this->checkContains({{0, 25}}, 10, true);
  // Check similar cases, but with larger sets.
  this->checkContains({{0, 5}, {10, 10}, {15, 20}}, 10, true);
  this->checkContains({{0, 5}, {10, 12}, {15, 20}}, 10, true);
  this->checkContains({{0, 5}, {5, 7}, {8, 10}, {12, 41}}, 10, true);

  constexpr TypeParam MIN = TestFixture::getMin();
  constexpr TypeParam MAX = TestFixture::getMax();
  constexpr TypeParam MID = TestFixture::getMid();
  this->checkContains({{MIN, MAX}}, 0, true);
  this->checkContains({{MIN, MAX}}, MID, true);
  this->checkContains({{MIN, MAX}}, -10, true);
  this->checkContains({{MIN, MAX}}, 10, true);
}

TYPED_TEST(RangeSetTest, RangeSetAddTest) {
  // Check adding single points
  this->checkAdd({}, 10, {{10, 10}});
  this->checkAdd({{0, 5}}, 10, {{0, 5}, {10, 10}});
  this->checkAdd({{0, 5}, {30, 40}}, 10, {{0, 5}, {10, 10}, {30, 40}});

  // Check adding single ranges.
  this->checkAdd({}, {10, 20}, {{10, 20}});
  this->checkAdd({{0, 5}}, {10, 20}, {{0, 5}, {10, 20}});
  this->checkAdd({{0, 5}, {30, 40}}, {10, 20}, {{0, 5}, {10, 20}, {30, 40}});

  // Check adding whole sets of ranges.
  this->checkAdd({{0, 5}}, {{10, 20}}, {{0, 5}, {10, 20}});
  // Check that ordering of ranges is as expected.
  this->checkAdd({{0, 5}, {30, 40}}, {{10, 20}}, {{0, 5}, {10, 20}, {30, 40}});
  this->checkAdd({{0, 5}, {30, 40}}, {{10, 20}, {50, 60}},
                 {{0, 5}, {10, 20}, {30, 40}, {50, 60}});
  this->checkAdd({{10, 20}, {50, 60}}, {{0, 5}, {30, 40}, {70, 80}},
                 {{0, 5}, {10, 20}, {30, 40}, {50, 60}, {70, 80}});
}

TYPED_TEST(RangeSetTest, RangeSetDeletePointTest) {
  constexpr TypeParam MIN = TestFixture::getMin();
  constexpr TypeParam MAX = TestFixture::getMax();
  constexpr TypeParam MID = TestFixture::getMid();

  this->checkDelete(MID, {{MIN, MAX}}, {{MIN, MID - 1}, {MID + 1, MAX}});
  // Check that delete works with an empty set.
  this->checkDelete(10, {}, {});
  // Check that delete can remove entire ranges.
  this->checkDelete(10, {{10, 10}}, {});
  this->checkDelete(10, {{0, 5}, {10, 10}, {20, 30}}, {{0, 5}, {20, 30}});
  // Check that delete can split existing ranges into two.
  this->checkDelete(10, {{0, 5}, {7, 15}, {20, 30}},
                    {{0, 5}, {7, 9}, {11, 15}, {20, 30}});
  // Check that delete of the point not from the range set works as expected.
  this->checkDelete(10, {{0, 5}, {20, 30}}, {{0, 5}, {20, 30}});
}
