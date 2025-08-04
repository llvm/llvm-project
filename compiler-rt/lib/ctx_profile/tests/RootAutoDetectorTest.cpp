#include "../RootAutoDetector.h"
#include "sanitizer_common/sanitizer_array_ref.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace __ctx_profile;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

// Utility for describing a preorder traversal. By default it captures the
// address and count at a callsite node. Implicitly nodes are expected to have 1
// child. If they have none, we place a Marker::term and if they have more than
// one, we place a Marker::split(nr_of_children) For example, using a list
// notation, and letters to denote a pair of address and count:
// (A (B C) (D (E F))) is a list of markers: A, split(2), B, term, C,
// term, D, split(2), E, term, F, term
class Marker {
  enum class Kind { End, Value, Split };
  const uptr Value;
  const uptr Count;
  const Kind K;
  Marker(uptr V, uptr C, Kind S) : Value(V), Count(C), K(S) {}

public:
  Marker(uptr V, uptr C) : Marker(V, C, Kind::Value) {}

  static Marker split(uptr V) { return Marker(V, 0, Kind::Split); }
  static Marker term() { return Marker(0, 0, Kind::End); }

  bool isSplit() const { return K == Kind::Split; }
  bool isTerm() const { return K == Kind::End; }
  bool isVal() const { return K == Kind::Value; }

  bool operator==(const Marker &M) const {
    return Value == M.Value && Count == M.Count && K == M.K;
  }
};

class MockCallsiteTrie final : public PerThreadCallsiteTrie {
  // Return the first multiple of 100.
  uptr getFctStartAddr(uptr CallsiteAddress) const override {
    return (CallsiteAddress / 100) * 100;
  }

  static void popAndCheck(ArrayRef<Marker> &Preorder, Marker M) {
    ASSERT_THAT(Preorder, Not(IsEmpty()));
    ASSERT_EQ(Preorder[0], M);
    Preorder = Preorder.drop_front();
  }

  static void checkSameImpl(const Trie &T, ArrayRef<Marker> &Preorder) {
    popAndCheck(Preorder, {T.CallsiteAddress, T.Count});

    if (T.Children.empty()) {
      popAndCheck(Preorder, Marker::term());
      return;
    }

    if (T.Children.size() > 1)
      popAndCheck(Preorder, Marker::split(T.Children.size()));

    T.Children.forEach([&](const auto &KVP) {
      checkSameImpl(KVP.second, Preorder);
      return true;
    });
  }

public:
  void checkSame(ArrayRef<Marker> Preorder) const {
    checkSameImpl(TheTrie, Preorder);
    ASSERT_THAT(Preorder, IsEmpty());
  }
};

TEST(PerThreadCallsiteTrieTest, Insert) {
  MockCallsiteTrie R;
  uptr Stack1[]{4, 3, 2, 1};
  R.insertStack(StackTrace(Stack1, 4));
  R.checkSame(ArrayRef<Marker>(
      {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, Marker::term()}));

  uptr Stack2[]{5, 4, 3, 2, 1};
  R.insertStack(StackTrace(Stack2, 5));
  R.checkSame(ArrayRef<Marker>(
      {{0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 1}, Marker::term()}));

  uptr Stack3[]{6, 3, 2, 1};
  R.insertStack(StackTrace(Stack3, 4));
  R.checkSame(ArrayRef<Marker>({{0, 3},
                                {1, 3},
                                {2, 3},
                                {3, 3},
                                Marker::split(2),
                                {4, 2},
                                {5, 1},
                                Marker::term(),
                                {6, 1},
                                Marker::term()}));
  uptr Stack4[]{7, 2, 1};
  R.insertStack(StackTrace(Stack4, 3));
  R.checkSame(ArrayRef<Marker>({{0, 4},
                                {1, 4},
                                {2, 4},
                                Marker::split(2),
                                {7, 1},
                                Marker::term(),
                                {3, 3},
                                Marker::split(2),
                                {4, 2},
                                {5, 1},
                                Marker::term(),
                                {6, 1},
                                Marker::term()}));
}

TEST(PerThreadCallsiteTrieTest, DetectRoots) {
  MockCallsiteTrie T;

  uptr Stack1[]{501, 302, 202, 102};
  uptr Stack2[]{601, 402, 203, 102};
  T.insertStack({Stack1, 4});
  T.insertStack({Stack2, 4});

  auto R = T.determineRoots();
  EXPECT_THAT(R, SizeIs(2U));
  EXPECT_TRUE(R.contains(300));
  EXPECT_TRUE(R.contains(400));
}

TEST(PerThreadCallsiteTrieTest, DetectRootsNoBranches) {
  MockCallsiteTrie T;

  uptr Stack1[]{501, 302, 202, 102};
  T.insertStack({Stack1, 4});

  auto R = T.determineRoots();
  EXPECT_THAT(R, IsEmpty());
}

TEST(PerThreadCallsiteTrieTest, DetectRootsUnknownFct) {
  MockCallsiteTrie T;

  uptr Stack1[]{501, 302, 202, 102};
  // The MockCallsiteTree address resolver resolves addresses over 100, so 40
  // will be mapped to 0.
  uptr Stack2[]{601, 40, 203, 102};
  T.insertStack({Stack1, 4});
  T.insertStack({Stack2, 4});

  auto R = T.determineRoots();
  ASSERT_THAT(R, SizeIs(2U));
  EXPECT_TRUE(R.contains(300));
  EXPECT_TRUE(R.contains(0));
}
