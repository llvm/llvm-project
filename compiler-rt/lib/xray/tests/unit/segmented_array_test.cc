#include "xray_segmented_array.h"
#include "gtest/gtest.h"

namespace __xray {
namespace {

struct TestData {
  s64 First;
  s64 Second;

  // Need a constructor for emplace operations.
  TestData(s64 F, s64 S) : First(F), Second(S) {}
};

TEST(SegmentedArrayTest, Construction) {
  Array<TestData> Data;
  (void)Data;
}

TEST(SegmentedArrayTest, ConstructWithAllocator) {
  using AllocatorType = typename Array<TestData>::AllocatorType;
  AllocatorType A(1 << 4, 0);
  Array<TestData> Data(A);
  (void)Data;
}

TEST(SegmentedArrayTest, ConstructAndPopulate) {
  Array<TestData> data;
  ASSERT_NE(data.Append(TestData{0, 0}), nullptr);
  ASSERT_NE(data.Append(TestData{1, 1}), nullptr);
  ASSERT_EQ(data.size(), 2u);
}

TEST(SegmentedArrayTest, ConstructPopulateAndLookup) {
  Array<TestData> data;
  ASSERT_NE(data.Append(TestData{0, 1}), nullptr);
  ASSERT_EQ(data.size(), 1u);
  ASSERT_EQ(data[0].First, 0);
  ASSERT_EQ(data[0].Second, 1);
}

TEST(SegmentedArrayTest, PopulateWithMoreElements) {
  Array<TestData> data;
  static const auto kMaxElements = 100u;
  for (auto I = 0u; I < kMaxElements; ++I) {
    ASSERT_NE(data.Append(TestData{I, I + 1}), nullptr);
  }
  ASSERT_EQ(data.size(), kMaxElements);
  for (auto I = 0u; I < kMaxElements; ++I) {
    ASSERT_EQ(data[I].First, I);
    ASSERT_EQ(data[I].Second, I + 1);
  }
}

TEST(SegmentedArrayTest, AppendEmplace) {
  Array<TestData> data;
  ASSERT_NE(data.AppendEmplace(1, 1), nullptr);
  ASSERT_EQ(data[0].First, 1);
  ASSERT_EQ(data[0].Second, 1);
}

TEST(SegmentedArrayTest, AppendAndTrim) {
  Array<TestData> data;
  ASSERT_NE(data.AppendEmplace(1, 1), nullptr);
  ASSERT_EQ(data.size(), 1u);
  data.trim(1);
  ASSERT_EQ(data.size(), 0u);
  ASSERT_TRUE(data.empty());
}

TEST(SegmentedArrayTest, IteratorAdvance) {
  Array<TestData> data;
  ASSERT_TRUE(data.empty());
  ASSERT_EQ(data.begin(), data.end());
  auto I0 = data.begin();
  ASSERT_EQ(I0++, data.begin());
  ASSERT_NE(I0, data.begin());
  for (const auto &D : data) {
    (void)D;
    FAIL();
  }
  ASSERT_NE(data.AppendEmplace(1, 1), nullptr);
  ASSERT_EQ(data.size(), 1u);
  ASSERT_NE(data.begin(), data.end());
  auto &D0 = *data.begin();
  ASSERT_EQ(D0.First, 1);
  ASSERT_EQ(D0.Second, 1);
}

TEST(SegmentedArrayTest, IteratorRetreat) {
  Array<TestData> data;
  ASSERT_TRUE(data.empty());
  ASSERT_EQ(data.begin(), data.end());
  ASSERT_NE(data.AppendEmplace(1, 1), nullptr);
  ASSERT_EQ(data.size(), 1u);
  ASSERT_NE(data.begin(), data.end());
  auto &D0 = *data.begin();
  ASSERT_EQ(D0.First, 1);
  ASSERT_EQ(D0.Second, 1);

  auto I0 = data.end();
  ASSERT_EQ(I0--, data.end());
  ASSERT_NE(I0, data.end());
  ASSERT_EQ(I0, data.begin());
  ASSERT_EQ(I0->First, 1);
  ASSERT_EQ(I0->Second, 1);
}

TEST(SegmentedArrayTest, IteratorTrimBehaviour) {
  using AllocatorType = typename Array<TestData>::AllocatorType;
  AllocatorType A(1 << 10, 0);
  Array<TestData> Data(A);
  ASSERT_TRUE(Data.empty());
  auto I0Begin = Data.begin(), I0End = Data.end();
  // Add enough elements in Data to have more than one chunk.
  constexpr auto Chunk = Array<TestData>::ChunkSize;
  constexpr auto ChunkX2 = Chunk * 2;
  for (auto i = ChunkX2; i > 0u; --i) {
    Data.AppendEmplace(static_cast<s64>(i), static_cast<s64>(i));
  }
  ASSERT_EQ(Data.size(), ChunkX2);
  {
    auto &Back = Data.back();
    ASSERT_EQ(Back.First, 1);
    ASSERT_EQ(Back.Second, 1);
  }

  // Trim one chunk's elements worth.
  Data.trim(Chunk);
  ASSERT_EQ(Data.size(), Chunk);

  // Check that we are still able to access 'back' properly.
  {
    auto &Back = Data.back();
    ASSERT_EQ(Back.First, static_cast<s64>(Chunk + 1));
    ASSERT_EQ(Back.Second, static_cast<s64>(Chunk + 1));
  }

  // Then trim until it's empty.
  Data.trim(Chunk);
  ASSERT_TRUE(Data.empty());

  // Here our iterators should be the same.
  auto I1Begin = Data.begin(), I1End = Data.end();
  EXPECT_EQ(I0Begin, I1Begin);
  EXPECT_EQ(I0End, I1End);

  // Then we ensure that adding elements back works just fine.
  for (auto i = ChunkX2; i > 0u; --i) {
    Data.AppendEmplace(static_cast<s64>(i), static_cast<s64>(i));
  }
  EXPECT_EQ(Data.size(), ChunkX2);
}

struct ShadowStackEntry {
  uint64_t EntryTSC = 0;
  uint64_t *NodePtr = nullptr;
  ShadowStackEntry(uint64_t T, uint64_t *N) : EntryTSC(T), NodePtr(N) {}
};

TEST(SegmentedArrayTest, SimulateStackBehaviour) {
  using AllocatorType = typename Array<ShadowStackEntry>::AllocatorType;
  AllocatorType A(1 << 10, 0);
  Array<ShadowStackEntry> Data(A);
  static uint64_t Dummy = 0;
  constexpr uint64_t Max = 9;

  for (uint64_t i = 0; i < Max; ++i) {
    auto P = Data.Append({i, &Dummy});
    ASSERT_NE(P, nullptr);
    ASSERT_EQ(P->NodePtr, &Dummy);
    auto &Back = Data.back();
    ASSERT_EQ(Back.NodePtr, &Dummy);
    ASSERT_EQ(Back.EntryTSC, i);
  }

  // Simulate a stack by checking the data from the end as we're trimming.
  auto Counter = Max;
  ASSERT_EQ(Data.size(), size_t(Max));
  while (!Data.empty()) {
    const auto &Top = Data.back();
    uint64_t *TopNode = Top.NodePtr;
    EXPECT_EQ(TopNode, &Dummy) << "Counter = " << Counter;
    Data.trim(1);
    --Counter;
    ASSERT_EQ(Data.size(), size_t(Counter));
  }
}

} // namespace
} // namespace __xray
