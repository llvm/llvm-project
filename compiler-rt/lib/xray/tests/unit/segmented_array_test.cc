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
  Array<TestData> data;
  ASSERT_TRUE(data.empty());
  auto I0Begin = data.begin(), I0End = data.end();
  // Add enough elements in data to have more than one chunk.
  constexpr auto ChunkX2 = Array<TestData>::ChunkSize * 2;
  for (auto i = ChunkX2; i > 0u; --i) {
    data.Append({static_cast<s64>(i), static_cast<s64>(i)});
  }
  ASSERT_EQ(data.size(), ChunkX2);
  // Trim one chunk's elements worth.
  data.trim(ChunkX2 / 2);
  ASSERT_EQ(data.size(), ChunkX2 / 2);
  // Then trim until it's empty.
  data.trim(ChunkX2 / 2);
  ASSERT_TRUE(data.empty());

  // Here our iterators should be the same.
  auto I1Begin = data.begin(), I1End = data.end();
  EXPECT_EQ(I0Begin, I1Begin);
  EXPECT_EQ(I0End, I1End);

  // Then we ensure that adding elements back works just fine.
  for (auto i = ChunkX2; i > 0u; --i) {
    data.Append({static_cast<s64>(i), static_cast<s64>(i)});
  }
  EXPECT_EQ(data.size(), ChunkX2);
}

} // namespace
} // namespace __xray
