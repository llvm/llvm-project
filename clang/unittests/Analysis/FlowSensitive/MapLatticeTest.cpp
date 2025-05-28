#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <ostream>

using namespace clang;
using namespace dataflow;

namespace {
// A simple lattice for basic tests.
class BooleanLattice : public DataflowLattice {
public:
  BooleanLattice() : Value(false) {}
  explicit BooleanLattice(bool B) : Value(B) {}

  static BooleanLattice bottom() { return BooleanLattice(false); }

  static BooleanLattice top() { return BooleanLattice(true); }

  DataflowLatticePtr clone() override {
    return std::make_unique<BooleanLattice>(*this);
  }

  bool isEqual(const DataflowLattice &Other) const override {
    return *this == llvm::cast<const BooleanLattice>(Other);
  }

  LatticeEffect join(const DataflowLattice &L) override {
    const auto &Other = llvm::cast<BooleanLattice>(L);
    auto Prev = Value;
    Value = Value || Other.Value;
    return Prev == Value ? LatticeJoinEffect::Unchanged
                         : LatticeJoinEffect::Changed;
  }

  friend bool operator==(BooleanLattice LHS, BooleanLattice RHS) {
    return LHS.Value == RHS.Value;
  }

  friend bool operator!=(BooleanLattice LHS, BooleanLattice RHS) {
    return LHS.Value != RHS.Value;
  }

  friend std::ostream &operator<<(std::ostream &Os, const BooleanLattice &B) {
    Os << B.Value;
    return Os;
  }

  bool value() const { return Value; }

private:
  bool Value;
};

using BoolMapLattice = MapLattice<int, BooleanLattice>;
} // namespace

static constexpr int Key1 = 0;
static constexpr int Key2 = 1;

namespace {
using ::testing::_;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(MapLatticeTest, InsertWorks) {
  BoolMapLattice Lattice;
  EXPECT_THAT(Lattice.insert({Key1, BooleanLattice(false)}), Pair(_, true));
  EXPECT_THAT(Lattice.insert({Key2, BooleanLattice(false)}), Pair(_, true));

  // Insertion fails on collision.
  EXPECT_THAT(Lattice.insert({Key1, BooleanLattice(false)}), Pair(_, false));
  EXPECT_THAT(Lattice.insert({Key2, BooleanLattice(false)}), Pair(_, false));

  EXPECT_THAT(Lattice, UnorderedElementsAre(Pair(Key1, BooleanLattice(false)),
                                            Pair(Key2, BooleanLattice(false))));
}

TEST(MapLatticeTest, ComparisonWorks) {
  BoolMapLattice Lattice1;
  Lattice1.insert({Key1, BooleanLattice(true)});
  Lattice1.insert({Key2, BooleanLattice(false)});
  BoolMapLattice Lattice2 = Lattice1;
  EXPECT_EQ(Lattice1, Lattice2);

  Lattice2.find(Key2)->second = BooleanLattice(true);
  EXPECT_NE(Lattice1, Lattice2);
}

TEST(MapLatticeTest, JoinChange) {
  BoolMapLattice Lattice1;
  Lattice1.insert({Key1, BooleanLattice(false)});
  Lattice1.insert({Key2, BooleanLattice(false)});

  BoolMapLattice Lattice2;
  Lattice2.insert({Key1, BooleanLattice(true)});
  Lattice2.insert({Key2, BooleanLattice(true)});

  ASSERT_THAT(Lattice1,
              UnorderedElementsAre(Pair(Key1, BooleanLattice(false)),
                                   Pair(Key2, BooleanLattice(false))));

  ASSERT_EQ(Lattice1.join(Lattice2), LatticeJoinEffect::Changed);
  EXPECT_THAT(Lattice1, UnorderedElementsAre(Pair(Key1, BooleanLattice(true)),
                                             Pair(Key2, BooleanLattice(true))));
}

TEST(MapLatticeTest, JoinEqNoChange) {
  BoolMapLattice Lattice;
  Lattice.insert({Key1, BooleanLattice(false)});
  Lattice.insert({Key2, BooleanLattice(false)});

  ASSERT_EQ(Lattice.join(Lattice), LatticeJoinEffect::Unchanged);
  EXPECT_THAT(Lattice, UnorderedElementsAre(Pair(Key1, BooleanLattice(false)),
                                            Pair(Key2, BooleanLattice(false))));
}

TEST(MapLatticeTest, JoinLtNoChange) {
  BoolMapLattice Lattice1;
  Lattice1.insert({Key1, BooleanLattice(false)});
  Lattice1.insert({Key2, BooleanLattice(false)});

  BoolMapLattice Lattice2;
  Lattice2.insert({Key1, BooleanLattice(true)});
  Lattice2.insert({Key2, BooleanLattice(true)});

  ASSERT_THAT(Lattice1,
              UnorderedElementsAre(Pair(Key1, BooleanLattice(false)),
                                   Pair(Key2, BooleanLattice(false))));

  ASSERT_THAT(Lattice2, UnorderedElementsAre(Pair(Key1, BooleanLattice(true)),
                                             Pair(Key2, BooleanLattice(true))));

  ASSERT_EQ(Lattice2.join(Lattice1), LatticeJoinEffect::Unchanged);
  EXPECT_THAT(Lattice2, UnorderedElementsAre(Pair(Key1, BooleanLattice(true)),
                                             Pair(Key2, BooleanLattice(true))));
}

TEST(MapLatticeTest, JoinDifferentDomainsProducesUnion) {
  BoolMapLattice Lattice1;
  Lattice1.insert({Key1, BooleanLattice(true)});
  BoolMapLattice Lattice2;
  Lattice2.insert({Key2, BooleanLattice(true)});

  ASSERT_EQ(Lattice1.join(Lattice2), LatticeJoinEffect::Changed);
  EXPECT_THAT(Lattice1, UnorderedElementsAre(Pair(Key1, BooleanLattice(true)),
                                             Pair(Key2, BooleanLattice(true))));
}

TEST(MapLatticeTest, FindWorks) {
  BoolMapLattice Lattice;
  Lattice.insert({Key1, BooleanLattice(true)});
  Lattice.insert({Key2, BooleanLattice(false)});

  auto It = Lattice.find(Key1);
  ASSERT_NE(It, Lattice.end());
  EXPECT_EQ(It->second, BooleanLattice(true));

  It = Lattice.find(Key2);
  ASSERT_NE(It, Lattice.end());
  EXPECT_EQ(It->second, BooleanLattice(false));
}

TEST(MapLatticeTest, ContainsWorks) {
  BoolMapLattice Lattice;
  Lattice.insert({Key1, BooleanLattice(true)});
  EXPECT_TRUE(Lattice.contains(Key1));
  EXPECT_FALSE(Lattice.contains(Key2));
}
} // namespace
