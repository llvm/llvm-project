#include "memprof/memprof_rawprofile.h"

#include <cstdint>
#include <memory>

#include "profile/MemProfData.inc"
#include "sanitizer_common/sanitizer_array_ref.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_vector.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::__memprof::MIBMapTy;
using ::__memprof::SerializeToRawProfile;
using ::__sanitizer::StackDepotPut;
using ::__sanitizer::StackTrace;
using ::llvm::memprof::MemInfoBlock;

uint64_t PopulateFakeMap(const MemInfoBlock &FakeMIB, uintptr_t StackPCBegin,
                         MIBMapTy &FakeMap) {
  constexpr int kSize = 5;
  uintptr_t array[kSize];
  for (int i = 0; i < kSize; i++) {
    array[i] = StackPCBegin + i;
  }
  StackTrace St(array, kSize);
  uint32_t Id = StackDepotPut(St);

  InsertOrMerge(Id, FakeMIB, FakeMap);
  return Id;
}

template <class T = uint64_t> T Read(char *&Buffer) {
  static_assert(std::is_pod<T>::value, "Must be a POD type.");
  assert(reinterpret_cast<size_t>(Buffer) % sizeof(T) == 0 &&
         "Unaligned read!");
  T t = *reinterpret_cast<T *>(Buffer);
  Buffer += sizeof(T);
  return t;
}

TEST(MemProf, Basic) {
  __sanitizer::LoadedModule FakeModule;
  FakeModule.addAddressRange(/*begin=*/0x10, /*end=*/0x20, /*executable=*/true,
                             /*writable=*/false, /*name=*/"");
  const char uuid[__sanitizer::kModuleUUIDSize] = {0xC, 0x0, 0xF,
                                                   0xF, 0xE, 0xE};
  FakeModule.setUuid(uuid, __sanitizer::kModuleUUIDSize);
  __sanitizer::ArrayRef<__sanitizer::LoadedModule> Modules(&FakeModule,
                                                           (&FakeModule) + 1);

  MIBMapTy FakeMap;
  MemInfoBlock FakeMIB;
  // Since we want to override the constructor set vals to make it easier to
  // test.
  memset(&FakeMIB, 0, sizeof(MemInfoBlock));
  FakeMIB.AllocCount = 0x1;
  FakeMIB.TotalAccessCount = 0x2;

  // Use large enough PC values to avoid underflow when
  // GetPreviousInstructionPc subtracts an architecture-dependent value.
  uint64_t FakeIds[2];
  FakeIds[0] = PopulateFakeMap(FakeMIB, /*StackPCBegin=*/100, FakeMap);
  FakeIds[1] = PopulateFakeMap(FakeMIB, /*StackPCBegin=*/200, FakeMap);

  char *Ptr = nullptr;
  __sanitizer::Vector<__sanitizer::u64> FakeAddresses;
  FakeAddresses.PushBack(0x100);
  FakeAddresses.PushBack(0x200);
  uint64_t NumBytes =
      SerializeToRawProfile(FakeMap, Modules, FakeAddresses, Ptr);
  const char *Buffer = Ptr;

  ASSERT_GT(NumBytes, 0ULL);
  ASSERT_TRUE(Ptr);

  // Check the header.
  EXPECT_THAT(Read(Ptr), MEMPROF_RAW_MAGIC_64);
  EXPECT_THAT(Read(Ptr), MEMPROF_RAW_VERSION);
  const uint64_t TotalSize = Read(Ptr);
  const uint64_t SegmentOffset = Read(Ptr);
  const uint64_t MIBOffset = Read(Ptr);
  const uint64_t StackOffset = Read(Ptr);
  const uint64_t MemAddressOffset = Read(Ptr);

  // ============= Check sizes and padding.
  EXPECT_EQ(TotalSize, NumBytes);
  EXPECT_EQ(TotalSize % 8, 0ULL);

  // Should be equal to the size of the raw profile header (7 fields * 8 bytes).
  EXPECT_EQ(SegmentOffset, 56ULL);

  // We expect only 1 segment entry, 8b for the count and 64b for SegmentEntry
  // in memprof_rawprofile.cpp.
  EXPECT_EQ(MIBOffset - SegmentOffset, 72ULL);

  EXPECT_EQ(MIBOffset, 128ULL);
  // We expect 2 mib entry, 8b for the count and sizeof(uint64_t) +
  // sizeof(MemInfoBlock) contains stack id + MeminfoBlock.
  EXPECT_EQ(StackOffset - MIBOffset, 8 + 2 * (8 + sizeof(MemInfoBlock)));

  EXPECT_EQ(StackOffset, 440ULL);
  // We expect 2 stack entries, with 5 frames - 8b for total count,
  // 2 * (8b for id, 8b for frame count and 5*8b for fake frames).
  // Since this is no longer the last section, check the exact size.
  EXPECT_GE(MemAddressOffset - StackOffset, 8ULL + 2 * (8 + 8 + 5 * 8));

  // We expect 2 address entries: 8b for count + 2 * 8b for addresses.
  EXPECT_GE(TotalSize - MemAddressOffset, 8ULL + 2 * 8);

  // ============= Check contents.
  // Build expected segment bytes dynamically since uuid size varies by
  // platform (kModuleUUIDSize is 16 on Apple, 32 on Linux).
  unsigned char ExpectedSegmentBytes[72] = {};
  // Number of entries = 1
  ExpectedSegmentBytes[0] = 0x01;
  // Start = 0x10
  ExpectedSegmentBytes[8] = 0x10;
  // End = 0x20
  ExpectedSegmentBytes[16] = 0x20;
  // Offset = 0x0 (base_address_ is 0 for default-constructed LoadedModule)
  // BuildIdSize = kModuleUUIDSize
  ExpectedSegmentBytes[32] =
      static_cast<unsigned char>(__sanitizer::kModuleUUIDSize);
  // Uuid
  ExpectedSegmentBytes[40] = 0xC;
  ExpectedSegmentBytes[41] = 0x0;
  ExpectedSegmentBytes[42] = 0xF;
  ExpectedSegmentBytes[43] = 0xF;
  ExpectedSegmentBytes[44] = 0xE;
  ExpectedSegmentBytes[45] = 0xE;
  EXPECT_EQ(memcmp(Buffer + SegmentOffset, ExpectedSegmentBytes, 72), 0);

  // Check that the number of entries is 2.
  EXPECT_EQ(*reinterpret_cast<const uint64_t *>(Buffer + MIBOffset), 2ULL);
  // Check that stack id is set.
  EXPECT_EQ(*reinterpret_cast<const uint64_t *>(Buffer + MIBOffset + 8),
            FakeIds[0]);

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
  EXPECT_EQ(*reinterpret_cast<const uint64_t *>(Buffer + StackOffset), 2ULL);
  // Check that the 1st stack id is set.
  EXPECT_EQ(*reinterpret_cast<const uint64_t *>(Buffer + StackOffset + 8),
            FakeIds[0]);
  // Build expected stack bytes dynamically since GetPreviousInstructionPc
  // applies an architecture-dependent adjustment (e.g., -1 on x86_64, -4 on
  // arm64).
  unsigned char ExpectedStackBytes[2][6 * 8] = {};
  for (int s = 0; s < 2; s++) {
    uintptr_t StackPCBegin = (s == 0) ? 100 : 200;
    // Number of PCs = 5
    ExpectedStackBytes[s][0] = 0x5;
    for (int i = 0; i < 5; i++) {
      uint64_t pc = StackTrace::GetPreviousInstructionPc(StackPCBegin + i);
      memcpy(&ExpectedStackBytes[s][(i + 1) * 8], &pc, sizeof(pc));
    }
  }
  EXPECT_EQ(memcmp(Buffer + StackOffset + 16, ExpectedStackBytes[0],
                   sizeof(ExpectedStackBytes[0])),
            0);

  // Check that the 2nd stack id is set.
  EXPECT_EQ(
      *reinterpret_cast<const uint64_t *>(Buffer + StackOffset + 8 + 6 * 8 + 8),
      FakeIds[1]);

  EXPECT_EQ(memcmp(Buffer + StackOffset + 16 + 6 * 8 + 8, ExpectedStackBytes[1],
                   sizeof(ExpectedStackBytes[1])),
            0);
}
} // namespace
