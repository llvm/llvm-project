//===-- chunk.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_CHUNK_H_
#define SCUDO_CHUNK_H_

#include "platform.h"

#include "atomic_helpers.h"
#include "checksum.h"
#include "common.h"
#include "report.h"

namespace scudo {

extern Checksum HashAlgorithm;

INLINE u16 computeChecksum(u32 Seed, uptr Value, uptr *Array, uptr ArraySize) {
  // If the hardware CRC32 feature is defined here, it was enabled everywhere,
  // as opposed to only for crc32_hw.cc. This means that other hardware specific
  // instructions were likely emitted at other places, and as a result there is
  // no reason to not use it here.
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
  u32 Crc = static_cast<u32>(CRC32_INTRINSIC(Seed, Value));
  for (uptr I = 0; I < ArraySize; I++)
    Crc = static_cast<u32>(CRC32_INTRINSIC(Crc, Array[I]));
  return static_cast<u16>((Crc & 0xffff) ^ (Crc >> 16));
#else
  if (HashAlgorithm == Checksum::HardwareCRC32) {
    u32 Crc = computeHardwareCRC32(Seed, Value);
    for (uptr I = 0; I < ArraySize; I++)
      Crc = computeHardwareCRC32(Crc, Array[I]);
    return static_cast<u16>((Crc & 0xffff) ^ (Crc >> 16));
  } else {
    u16 Checksum = computeBSDChecksum(static_cast<u16>(Seed & 0xffff), Value);
    for (uptr I = 0; I < ArraySize; I++)
      Checksum = computeBSDChecksum(Checksum, Array[I]);
    return Checksum;
  }
#endif // defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
}

namespace Chunk {

// Note that in an ideal world, `State` and `Origin` should be `enum class`, and
// the associated `UnpackedHeader` fields of their respective enum class type
// but https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61414 prevents it from
// happening, as it will error, complaining the number of bits is not enough.
enum Origin : u8 {
  Malloc = 0,
  New = 1,
  NewArray = 2,
  Memalign = 3,
};

enum State : u8 { Available = 0, Allocated = 1, Quarantined = 2 };

typedef u64 PackedHeader;
// Update the 'Mask' constants to reflect changes in this structure.
struct UnpackedHeader {
  u64 Checksum : 16;
  u64 ClassId : 8;
  u64 SizeOrUnusedBytes : 20;
  u8 State : 2;
  u8 Origin : 2;
  u64 Offset : 16;
};
typedef atomic_u64 AtomicPackedHeader;
COMPILER_CHECK(sizeof(UnpackedHeader) == sizeof(PackedHeader));

// Those constants are required to silence some -Werror=conversion errors when
// assigning values to the related bitfield variables.
constexpr uptr ChecksumMask = (1UL << 16) - 1;
constexpr uptr ClassIdMask = (1UL << 8) - 1;
constexpr uptr SizeOrUnusedBytesMask = (1UL << 20) - 1;
constexpr uptr StateMask = (1UL << 2) - 1;
constexpr uptr OriginMask = (1UL << 2) - 1;
constexpr uptr OffsetMask = (1UL << 16) - 1;

constexpr uptr getHeaderSize() {
  return roundUpTo(sizeof(PackedHeader), 1U << SCUDO_MIN_ALIGNMENT_LOG);
}

INLINE AtomicPackedHeader *getAtomicHeader(void *Ptr) {
  return reinterpret_cast<AtomicPackedHeader *>(reinterpret_cast<uptr>(Ptr) -
                                                getHeaderSize());
}

INLINE
const AtomicPackedHeader *getConstAtomicHeader(const void *Ptr) {
  return reinterpret_cast<const AtomicPackedHeader *>(
      reinterpret_cast<uptr>(Ptr) - getHeaderSize());
}

// We do not need a cryptographically strong hash for the checksum, but a CRC
// type function that can alert us in the event a header is invalid or
// corrupted. Ideally slightly better than a simple xor of all fields.
static INLINE u16 computeHeaderChecksum(u32 Cookie, const void *Ptr,
                                        UnpackedHeader *Header) {
  UnpackedHeader ZeroChecksumHeader = *Header;
  ZeroChecksumHeader.Checksum = 0;
  uptr HeaderHolder[sizeof(UnpackedHeader) / sizeof(uptr)];
  memcpy(&HeaderHolder, &ZeroChecksumHeader, sizeof(HeaderHolder));
  return computeChecksum(Cookie, reinterpret_cast<uptr>(Ptr), HeaderHolder,
                         ARRAY_SIZE(HeaderHolder));
}

INLINE void storeHeader(u32 Cookie, void *Ptr,
                        UnpackedHeader *NewUnpackedHeader) {
  NewUnpackedHeader->Checksum =
      computeHeaderChecksum(Cookie, Ptr, NewUnpackedHeader);
  PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
  atomic_store_relaxed(getAtomicHeader(Ptr), NewPackedHeader);
}

INLINE
void loadHeader(u32 Cookie, const void *Ptr,
                UnpackedHeader *NewUnpackedHeader) {
  PackedHeader NewPackedHeader = atomic_load_relaxed(getConstAtomicHeader(Ptr));
  *NewUnpackedHeader = bit_cast<UnpackedHeader>(NewPackedHeader);
  if (UNLIKELY(NewUnpackedHeader->Checksum !=
               computeHeaderChecksum(Cookie, Ptr, NewUnpackedHeader)))
    reportHeaderCorruption(const_cast<void *>(Ptr));
}

INLINE void compareExchangeHeader(u32 Cookie, void *Ptr,
                                  UnpackedHeader *NewUnpackedHeader,
                                  UnpackedHeader *OldUnpackedHeader) {
  NewUnpackedHeader->Checksum =
      computeHeaderChecksum(Cookie, Ptr, NewUnpackedHeader);
  PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
  PackedHeader OldPackedHeader = bit_cast<PackedHeader>(*OldUnpackedHeader);
  if (UNLIKELY(!atomic_compare_exchange_strong(
          getAtomicHeader(Ptr), &OldPackedHeader, NewPackedHeader,
          memory_order_relaxed)))
    reportHeaderRace(Ptr);
}

INLINE
bool isValid(u32 Cookie, const void *Ptr, UnpackedHeader *NewUnpackedHeader) {
  PackedHeader NewPackedHeader = atomic_load_relaxed(getConstAtomicHeader(Ptr));
  *NewUnpackedHeader = bit_cast<UnpackedHeader>(NewPackedHeader);
  return NewUnpackedHeader->Checksum ==
         computeHeaderChecksum(Cookie, Ptr, NewUnpackedHeader);
}

} // namespace Chunk

} // namespace scudo

#endif // SCUDO_CHUNK_H_
