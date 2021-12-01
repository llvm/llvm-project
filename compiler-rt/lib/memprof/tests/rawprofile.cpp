#include "memprof/memprof_rawprofile.h"

#include <cstdint>
#include <memory>

#include "memprof/memprof_meminfoblock.h"
#include "profile/MemProfData.inc"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::__memprof::MemInfoBlock;
using ::__memprof::MIBMapTy;
using ::__memprof::SerializeToRawProfile;
using ::__sanitizer::MemoryMappedSegment;
using ::__sanitizer::MemoryMappingLayoutBase;
using ::__sanitizer::StackDepotPut;
using ::__sanitizer::StackTrace;
using ::testing::_;
using ::testing::Action;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class MockMemoryMappingLayout final : public MemoryMappingLayoutBase {
public:
  MOCK_METHOD(bool, Next, (MemoryMappedSegment *), (override));
  MOCK_METHOD(void, Reset, (), (override));
};

u64 PopulateFakeMap(const MemInfoBlock &FakeMIB, uptr StackPCBegin,
                    MIBMapTy &FakeMap) {
  constexpr int kSize = 5;
  uptr array[kSize];
  for (int i = 0; i < kSize; i++) {
    array[i] = StackPCBegin + i;
  }
  StackTrace St(array, kSize);
  u32 Id = StackDepotPut(St);

  InsertOrMerge(Id, FakeMIB, FakeMap);
  return Id;
}

template <class T = u64> T Read(char *&Buffer) {
  static_assert(std::is_pod<T>::value, "Must be a POD type.");
  assert(reinterpret_cast<size_t>(Buffer) % sizeof(T) == 0 &&
         "Unaligned read!");
  T t = *reinterpret_cast<T *>(Buffer);
  Buffer += sizeof(T);
  return t;
}

TEST(MemProf, Basic) {
  MockMemoryMappingLayout Layout;
  MemoryMappedSegment FakeSegment;
  memset(&FakeSegment, 0, sizeof(FakeSegment));
  FakeSegment.start = 0x10;
  FakeSegment.end = 0x20;
  FakeSegment.offset = 0x10;
  uint8_t uuid[__sanitizer::kModuleUUIDSize] = {0xC, 0x0, 0xF, 0xF, 0xE, 0xE};
  memcpy(FakeSegment.uuid, uuid, __sanitizer::kModuleUUIDSize);
  FakeSegment.protection =
      __sanitizer::kProtectionExecute | __sanitizer::kProtectionRead;

  const Action<bool(MemoryMappedSegment *)> SetSegment =
      DoAll(SetArgPointee<0>(FakeSegment), Return(true));
  EXPECT_CALL(Layout, Next(_))
      .WillOnce(SetSegment)
      .WillOnce(Return(false))
      .WillOnce(SetSegment)
      .WillRepeatedly(Return(false));

  EXPECT_CALL(Layout, Reset).Times(2);

  MIBMapTy FakeMap;
  MemInfoBlock FakeMIB;
  // Since we want to override the constructor set vals to make it easier to
  // test.
  memset(&FakeMIB, 0, sizeof(MemInfoBlock));
  FakeMIB.alloc_count = 0x1;
  FakeMIB.total_access_count = 0x2;

  u64 FakeIds[2];
  FakeIds[0] = PopulateFakeMap(FakeMIB, /*StackPCBegin=*/2, FakeMap);
  FakeIds[1] = PopulateFakeMap(FakeMIB, /*StackPCBegin=*/3, FakeMap);

  char *Ptr = nullptr;
  u64 NumBytes = SerializeToRawProfile(FakeMap, Layout, Ptr);
  const char *Buffer = Ptr;

  ASSERT_GT(NumBytes, 0ULL);
  ASSERT_TRUE(Ptr);

  // Check the header.
  EXPECT_THAT(Read(Ptr), MEMPROF_RAW_MAGIC_64);
  EXPECT_THAT(Read(Ptr), MEMPROF_RAW_VERSION);
  const u64 TotalSize = Read(Ptr);
  const u64 SegmentOffset = Read(Ptr);
  const u64 MIBOffset = Read(Ptr);
  const u64 StackOffset = Read(Ptr);

  // ============= Check sizes and padding.
  EXPECT_EQ(TotalSize, NumBytes);
  EXPECT_EQ(TotalSize % 8, 0ULL);

  // Should be equal to the size of the raw profile header.
  EXPECT_EQ(SegmentOffset, 48ULL);

  // We expect only 1 segment entry, 8b for the count and 56b for SegmentEntry
  // in memprof_rawprofile.cpp.
  EXPECT_EQ(MIBOffset - SegmentOffset, 64ULL);

  EXPECT_EQ(MIBOffset, 112ULL);
  // We expect 2 mib entry, 8b for the count and sizeof(u64) +
  // sizeof(MemInfoBlock) contains stack id + MeminfoBlock.
  EXPECT_EQ(StackOffset - MIBOffset, 8 + 2 * (8 + sizeof(MemInfoBlock)));

  EXPECT_EQ(StackOffset, 336ULL);
  // We expect 2 stack entries, with 5 frames - 8b for total count,
  // 2 * (8b for id, 8b for frame count and 5*8b for fake frames).
  // Since this is the last section, there may be additional padding at the end
  // to make the total profile size 8b aligned.
  EXPECT_GE(TotalSize - StackOffset, 8ULL + 2 * (8 + 8 + 5 * 8));

  // ============= Check contents.
  unsigned char ExpectedSegmentBytes[64] = {
      0x01, 0,   0,   0,   0,   0,   0, 0, // Number of entries
      0x10, 0,   0,   0,   0,   0,   0, 0, // Start
      0x20, 0,   0,   0,   0,   0,   0, 0, // End
      0x10, 0,   0,   0,   0,   0,   0, 0, // Offset
      0x0C, 0x0, 0xF, 0xF, 0xE, 0xE,       // Uuid
  };
  EXPECT_EQ(memcmp(Buffer + SegmentOffset, ExpectedSegmentBytes, 64), 0);

  // Check that the number of entries is 2.
  EXPECT_EQ(*reinterpret_cast<const u64 *>(Buffer + MIBOffset), 2ULL);
  // Check that stack id is set.
  EXPECT_EQ(*reinterpret_cast<const u64 *>(Buffer + MIBOffset + 8), FakeIds[0]);

  // Only check a few fields of the first MemInfoBlock.
  unsigned char ExpectedMIBBytes[sizeof(MemInfoBlock)] = {
      0x01, 0, 0, 0, // Alloc count
      0x02, 0, 0, 0, // Total access count
  };
  // Compare contents of 1st MIB after skipping count and stack id.
  EXPECT_EQ(
      memcmp(Buffer + MIBOffset + 16, ExpectedMIBBytes, sizeof(MemInfoBlock)),
      0);
  // Compare contents of 2nd MIB after skipping count and stack id for the first
  // and only the id for the second.
  EXPECT_EQ(memcmp(Buffer + MIBOffset + 16 + sizeof(MemInfoBlock) + 8,
                   ExpectedMIBBytes, sizeof(MemInfoBlock)),
            0);

  // Check that the number of entries is 2.
  EXPECT_EQ(*reinterpret_cast<const u64 *>(Buffer + StackOffset), 2ULL);
  // Check that the 1st stack id is set.
  EXPECT_EQ(*reinterpret_cast<const u64 *>(Buffer + StackOffset + 8),
            FakeIds[0]);
  // Contents are num pcs, value of each pc - 1.
  unsigned char ExpectedStackBytes[2][6 * 8] = {
      {
          0x5, 0, 0, 0, 0, 0, 0, 0, // Number of PCs
          0x1, 0, 0, 0, 0, 0, 0, 0, // PC ...
          0x2, 0, 0, 0, 0, 0, 0, 0, 0x3, 0, 0, 0, 0, 0, 0, 0,
          0x4, 0, 0, 0, 0, 0, 0, 0, 0x5, 0, 0, 0, 0, 0, 0, 0,
      },
      {
          0x5, 0, 0, 0, 0, 0, 0, 0, // Number of PCs
          0x2, 0, 0, 0, 0, 0, 0, 0, // PC ...
          0x3, 0, 0, 0, 0, 0, 0, 0, 0x4, 0, 0, 0, 0, 0, 0, 0,
          0x5, 0, 0, 0, 0, 0, 0, 0, 0x6, 0, 0, 0, 0, 0, 0, 0,
      },
  };
  EXPECT_EQ(memcmp(Buffer + StackOffset + 16, ExpectedStackBytes[0],
                   sizeof(ExpectedStackBytes[0])),
            0);

  // Check that the 2nd stack id is set.
  EXPECT_EQ(
      *reinterpret_cast<const u64 *>(Buffer + StackOffset + 8 + 6 * 8 + 8),
      FakeIds[1]);

  EXPECT_EQ(memcmp(Buffer + StackOffset + 16 + 6 * 8 + 8, ExpectedStackBytes[1],
                   sizeof(ExpectedStackBytes[1])),
            0);
}

} // namespace
