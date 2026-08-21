//===-- flang/unittests/Evaluate/LogicalValueTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/logical-value.h"
#include <cstddef>
#include <iterator>
#include <utility>

using namespace Fortran::common;
using namespace Fortran::evaluate;
using namespace Fortran::evaluate::value;

namespace {

class LogicalValueKind : public testing::TestWithParam<KindsEnum> {};
INSTANTIATE_TEST_SUITE_P(LogicalValueKind, LogicalValueKind,
    testing::ValuesIn(LogicalKinds),
    [](const testing::TestParamInfo<KindsEnum> &info) {
      return "LOGICAL(" + std::to_string(static_cast<int>(info.param)) + ")";
    });

constexpr int KindPos(KindsEnum kind) {
  for (std::size_t i{0}; i < std::size(LogicalKinds); ++i) {
    if (LogicalKinds[i] == kind) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(LogicalValue, DefaultConstructionIsMonostate) {
  LogicalValue x;
  EXPECT_TRUE(x.IsMonostate());
  EXPECT_FALSE(x.IsTrue());
}

TEST_P(LogicalValueKind, ConstructFromBool) {
  const KindsEnum kind{GetParam()};

  LogicalValue truth{kind, true};
  EXPECT_FALSE(truth.IsMonostate());
  EXPECT_EQ(kind, truth.kind());
  EXPECT_TRUE(truth.IsTrue());

  LogicalValue falsehood{kind, false};
  EXPECT_EQ(kind, falsehood.kind());
  EXPECT_FALSE(falsehood.IsTrue());
}

TEST_P(LogicalValueKind, ConstructFromWord) {
  const KindsEnum kind{GetParam()};

  LogicalValue zero{kind, IntegerValue{kind, 0}};
  EXPECT_FALSE(zero.IsTrue());

  LogicalValue one{kind, IntegerValue{kind, 1}};
  EXPECT_TRUE(one.IsTrue());

  LogicalValue two{kind, IntegerValue{kind, 2}};
  EXPECT_TRUE(two.IsTrue());

  LogicalValue allOnes{kind, IntegerValue{kind, -1}};
  EXPECT_TRUE(allOnes.IsTrue());
}

TEST_P(LogicalValueKind, CopyAndMove) {
  const KindsEnum kind{GetParam()};

  LogicalValue x{kind, true};
  LogicalValue copyConstructed{x};
  EXPECT_TRUE(copyConstructed.IsTrue());

  LogicalValue copyAssigned;
  copyAssigned = x;
  EXPECT_EQ(kind, copyAssigned.kind());
  EXPECT_TRUE(copyAssigned.IsTrue());

  LogicalValue moveConstructed{std::move(copyConstructed)};
  EXPECT_TRUE(moveConstructed.IsTrue());

  LogicalValue moveAssigned;
  moveAssigned = std::move(copyAssigned);
  EXPECT_TRUE(moveAssigned.IsTrue());
}

TEST_P(LogicalValueKind, KindCheckingConstructors) {
  const KindsEnum kind{GetParam()};

  LogicalValue x{kind, true};
  EXPECT_EQ(kind, LogicalValue(kind, x).kind());
  EXPECT_TRUE(LogicalValue(kind, x).IsTrue());

  LogicalValue y{kind, false};
  LogicalValue moved{kind, std::move(y)};
  EXPECT_EQ(kind, moved.kind());
  EXPECT_FALSE(moved.IsTrue());
}

TEST_P(LogicalValueKind, Zero) {
  const KindsEnum kind{GetParam()};

  LogicalValue zero{LogicalValue::Zero(kind)};
  EXPECT_FALSE(zero.IsMonostate());
  EXPECT_EQ(kind, zero.kind());
  EXPECT_FALSE(zero.IsTrue());
  EXPECT_TRUE(zero.IsCanonical());
}

TEST(LogicalValue, Bits) {
  EXPECT_EQ(8, LogicalValue::bits(KindsEnum::Kind1));
  EXPECT_EQ(16, LogicalValue::bits(KindsEnum::Kind2));
  EXPECT_EQ(32, LogicalValue::bits(KindsEnum::Kind4));
  EXPECT_EQ(64, LogicalValue::bits(KindsEnum::Kind8));
  EXPECT_EQ(32, LogicalValue(KindsEnum::Kind4, true).bits());
}

TEST(LogicalValue, BytesStored) {
  EXPECT_EQ(1u, LogicalValue::bytesStored(KindsEnum::Kind1));
  EXPECT_EQ(2u, LogicalValue::bytesStored(KindsEnum::Kind2));
  EXPECT_EQ(4u, LogicalValue::bytesStored(KindsEnum::Kind4));
  EXPECT_EQ(8u, LogicalValue::bytesStored(KindsEnum::Kind8));
  EXPECT_EQ(4u, LogicalValue(KindsEnum::Kind4, true).bytesStored());
}

TEST_P(LogicalValueKind, Word_) {
  const KindsEnum kind{GetParam()};

  // .TRUE. is represented canonically as 1 and .FALSE. as 0.
  EXPECT_EQ(1, LogicalValue(kind, true).word().ToInt64());
  EXPECT_EQ(0, LogicalValue(kind, false).word().ToInt64());
  EXPECT_EQ(kind, LogicalValue(kind, true).word().kind());
  // A word constructed from a raw pattern is preserved.
  EXPECT_EQ(2, LogicalValue(kind, IntegerValue{kind, 2}).word().ToInt64());
}

TEST_P(LogicalValueKind, IsCanonical) {
  const KindsEnum kind{GetParam()};

  EXPECT_TRUE(LogicalValue(kind, true).IsCanonical());
  EXPECT_TRUE(LogicalValue(kind, false).IsCanonical());
  EXPECT_TRUE(LogicalValue(kind, IntegerValue{kind, 0}).IsCanonical());
  EXPECT_TRUE(LogicalValue(kind, IntegerValue{kind, 1}).IsCanonical());
  EXPECT_FALSE(LogicalValue(kind, IntegerValue{kind, 2}).IsCanonical());
  EXPECT_FALSE(LogicalValue(kind, IntegerValue{kind, -1}).IsCanonical());
}

TEST_P(LogicalValueKind, IsTrue) {
  const KindsEnum kind{GetParam()};

  EXPECT_FALSE(LogicalValue{}.IsTrue());
  EXPECT_FALSE(LogicalValue(kind, false).IsTrue());
  EXPECT_TRUE(LogicalValue(kind, true).IsTrue());
  EXPECT_TRUE(LogicalValue(kind, IntegerValue{kind, 2}).IsTrue());
}

TEST_P(LogicalValueKind, RelationalOperators) {
  const KindsEnum kind{GetParam()};
  LogicalValue f{kind, false}, t{kind, true};

  EXPECT_TRUE(f < t);
  EXPECT_FALSE(t < f);
  EXPECT_FALSE(f < f);
  EXPECT_FALSE(t < t);

  EXPECT_TRUE(f <= f);
  EXPECT_TRUE(f <= t);
  EXPECT_FALSE(t <= f);
  EXPECT_FALSE(t <= t);

  EXPECT_TRUE(f == f);
  EXPECT_TRUE(t == t);
  EXPECT_FALSE(f == t);
  EXPECT_FALSE(f != f);
  EXPECT_TRUE(f != t);

  EXPECT_TRUE(t >= t);
  EXPECT_TRUE(t >= f);
  EXPECT_FALSE(f >= f);
  EXPECT_FALSE(f >= t);

  EXPECT_TRUE(t > f);
  EXPECT_FALSE(f > t);
  EXPECT_FALSE(t > t);
  EXPECT_FALSE(f > f);

  EXPECT_TRUE(LogicalValue(kind, IntegerValue{kind, 2}) == t);
}

TEST_P(LogicalValueKind, NOT) {
  const KindsEnum kind{GetParam()};

  EXPECT_TRUE(LogicalValue(kind, false).NOT().IsTrue());
  EXPECT_FALSE(LogicalValue(kind, true).NOT().IsTrue());
  EXPECT_EQ(kind, LogicalValue(kind, true).NOT().kind());
}

TEST_P(LogicalValueKind, AND) {
  const KindsEnum kind{GetParam()};

  LogicalValue f{kind, false}, t{kind, true};
  EXPECT_FALSE(f.AND(f).IsTrue());
  EXPECT_FALSE(f.AND(t).IsTrue());
  EXPECT_FALSE(t.AND(f).IsTrue());
  EXPECT_TRUE(t.AND(t).IsTrue());
  EXPECT_EQ(kind, t.AND(t).kind());
}

TEST_P(LogicalValueKind, OR) {
  const KindsEnum kind{GetParam()};

  LogicalValue f{kind, false}, t{kind, true};
  EXPECT_FALSE(f.OR(f).IsTrue());
  EXPECT_TRUE(f.OR(t).IsTrue());
  EXPECT_TRUE(t.OR(f).IsTrue());
  EXPECT_TRUE(t.OR(t).IsTrue());
  EXPECT_EQ(kind, f.OR(f).kind());
}

TEST_P(LogicalValueKind, EQV) {
  const KindsEnum kind{GetParam()};

  LogicalValue f{kind, false}, t{kind, true};
  EXPECT_TRUE(f.EQV(f).IsTrue());
  EXPECT_FALSE(f.EQV(t).IsTrue());
  EXPECT_FALSE(t.EQV(f).IsTrue());
  EXPECT_TRUE(t.EQV(t).IsTrue());
  EXPECT_EQ(kind, f.EQV(f).kind());
}

TEST_P(LogicalValueKind, NEQV) {
  const KindsEnum kind{GetParam()};

  LogicalValue f{kind, false}, t{kind, true};
  EXPECT_FALSE(f.NEQV(f).IsTrue());
  EXPECT_TRUE(f.NEQV(t).IsTrue());
  EXPECT_TRUE(t.NEQV(f).IsTrue());
  EXPECT_FALSE(t.NEQV(t).IsTrue());
  EXPECT_EQ(kind, f.NEQV(f).kind());
}

TEST_P(LogicalValueKind, RawBytesRoundTrip) {
  const KindsEnum kind{GetParam()};

  for (bool truth : {false, true}) {
    SCOPED_TRACE(testing::Message() << "truth=" << truth);

    LogicalValue original{kind, truth};
    char buffer[8]{};
    ASSERT_EQ(LogicalValue::bytesStored(kind), original.bytesStored())
        << "truth=" << truth;
    bool changed{false};
    original.StoreRawBytes(buffer, original.bytesStored(), &changed);
    EXPECT_EQ(truth, changed) << "truth=" << truth;
    LogicalValue restored{
        LogicalValue::FromRawBytes(kind, buffer, original.bytesStored())};
    EXPECT_EQ(kind, restored.kind()) << "truth=" << truth;
    EXPECT_EQ(truth, restored.IsTrue()) << "truth=" << truth;
    EXPECT_TRUE(restored.IsCanonical()) << "truth=" << truth;
  }
}

TEST_P(LogicalValueKind, Print) {
  const KindsEnum kind{GetParam()};
  const int pos{KindPos(kind)};

  struct Case {
    LogicalValue value;
    const char *results[4];
  };
  const Case cases[]{
      {LogicalValue{kind, false},
          {".false._1", ".false._2", ".false._4", ".false._8"}},
      {LogicalValue{kind, true},
          {".true._1", ".true._2", ".true._4", ".true._8"}},
      {LogicalValue{kind, IntegerValue{kind, 2}},
          {"transfer(2_1,.false._1)", "transfer(2_2,.false._2)",
              "transfer(2_4,.false._4)", "transfer(2_8,.false._8)"}},
  };

  for (const auto &c : cases) {
    llvm::SmallString<128> buf;
    llvm::raw_svector_ostream os{buf};
    c.value.print(os);
    EXPECT_EQ(c.results[pos], os.str());
  }
}

// Replicates the coverage of the legacy non-GTest test
// flang/unittests/Evaluate/logical.cpp.
TEST_P(LogicalValueKind, TruthTables) {
  const KindsEnum kind{GetParam()};

  EXPECT_EQ(8 * static_cast<int>(kind), LogicalValue::bits(kind));
  EXPECT_FALSE(LogicalValue{}.IsTrue());
  EXPECT_FALSE(LogicalValue(kind, false).IsTrue());
  EXPECT_TRUE(LogicalValue(kind, true).IsTrue());
  EXPECT_TRUE(LogicalValue(kind, false).NOT().IsTrue());
  EXPECT_FALSE(LogicalValue(kind, true).NOT().IsTrue());
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      LogicalValue a{kind, x}, b{kind, y};
      SCOPED_TRACE(
          testing::Message() << "kind=" << kind << " x=" << x << " y=" << y);

      EXPECT_EQ(x && y, a.AND(b).IsTrue());
      EXPECT_EQ(x || y, a.OR(b).IsTrue());
      EXPECT_EQ(x == y, a.EQV(b).IsTrue());
      EXPECT_EQ(x != y, a.NEQV(b).IsTrue());
    }
  }
}

} // namespace
