//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Configurable packed watchpoint table shared by the host and GPU runtimes.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_WATCH_H
#define CSAN_WATCH_H

#include "csan_defs.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

#if !defined(__has_builtin) || !__has_builtin(__scoped_atomic_load_n)
#define CSAN_DEFINED_SCOPED_ATOMICS
#define __scoped_atomic_load_n(P, Order, Scope) __atomic_load_n(P, Order)
#define __scoped_atomic_store_n(P, V, Order, Scope)                            \
  __atomic_store_n(P, V, Order)
#define __scoped_atomic_compare_exchange_n(P, E, V, Weak, Success, Failure,    \
                                           Scope)                              \
  __atomic_compare_exchange_n(P, E, V, Weak, Success, Failure)
#define __scoped_atomic_exchange_n(P, V, Order, Scope)                         \
  __atomic_exchange_n(P, V, Order)
#endif

namespace __csan {

using __sanitizer::u32;
using __sanitizer::u64;
using __sanitizer::uptr;

template <u32 MaxAccessSize, u32 CheckAdjacentSlots> class WatchpointTable {
  static_assert(MaxAccessSize && !(MaxAccessSize & (MaxAccessSize - 1)),
                "maximum access size must be a power of two");

public:
  static constexpr u32 SizeBits = __builtin_popcount(MaxAccessSize - 1) + 1;
  static constexpr u32 AddressBits = 64 - 2 - SizeBits;
  static constexpr u64 AddressMask = (1ull << AddressBits) - 1;
  static constexpr u32 OverflowEntries = 2 * CheckAdjacentSlots;

private:
  // Armed watchpoint (u64):
  //   [63]                 is_write
  //   [62]                 consumed = 0
  //   [61:AddressBits]     access size
  //   [AddressBits-1:0]    address key
  //
  // Consumed watchpoint (u64):
  //   [63]                 peer is_write
  //   [62]                 consumed = 1
  //   [61:AddressBits]     peer access size
  //   [AddressBits-1:0]    peer PC
  static constexpr u64 Invalid = 0;
  static constexpr u64 ConsumedMask = 1ull << 62;
  static constexpr u64 WriteMask = 1ull << 63;
  static constexpr u64 SizeMask = ((1ull << SizeBits) - 1) << AddressBits;
  static constexpr u32 NumSlots = 1 + 2 * CheckAdjacentSlots;

  u64 *Table;

  static constexpr u64 Encode(u64 Address, u32 Size, bool IsWrite) {
    return (IsWrite ? WriteMask : 0) | (static_cast<u64>(Size) << AddressBits) |
           (Address & AddressMask);
  }

  static constexpr bool Decode(u64 Value, u64 &Address, u32 &Size,
                               bool &IsWrite) {
    if (Value == Invalid || (Value & ConsumedMask))
      return false;
    IsWrite = Value & WriteMask;
    Size = (Value & SizeMask) >> AddressBits;
    Address = Value & AddressMask;
    return true;
  }

  static constexpr bool Overlaps(u64 A, u32 ASize, u64 B, u32 BSize) {
    return A < B + BSize && B < A + ASize;
  }

public:
  constexpr WatchpointTable(u64 *Table) : Table(Table) {}

  static constexpr u32 EncodeSize(u32 Size) {
    return Size < MaxAccessSize ? Size : MaxAccessSize;
  }

  u64 *Find(u64 Key, u32 Size, bool ExpectWrite, u32 Slot, u64 &Encoded) const {
    for (u32 I = 0; I < NumSlots; ++I) {
      u64 *Watchpoint = &Table[Slot + I];
      Encoded = __scoped_atomic_load_n(Watchpoint, __ATOMIC_RELAXED,
                                       __MEMORY_SCOPE_DEVICE);

      u64 Address;
      u32 WatchSize;
      bool IsWrite;
      if (!Decode(Encoded, Address, WatchSize, IsWrite))
        continue;
      if (ExpectWrite && !IsWrite)
        continue;
      if (Overlaps(Address, WatchSize, Key, Size))
        return Watchpoint;
    }
    return nullptr;
  }

  u64 *Insert(u64 Key, u32 Size, bool IsWrite, u32 Slot) const {
    const u64 Encoded = Encode(Key, Size, IsWrite);
    for (u32 I = 0; I < NumSlots; ++I) {
      u32 Index = Slot + ((I + CheckAdjacentSlots) % NumSlots);
      u64 *Watchpoint = &Table[Index];
      u64 Expected = Invalid;
      if (__scoped_atomic_compare_exchange_n(
              Watchpoint, &Expected, Encoded, false, __ATOMIC_RELAXED,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE))
        return Watchpoint;
    }
    return nullptr;
  }

  bool TryConsume(u64 *Watchpoint, u64 Encoded, uptr PC, bool IsWrite,
                  u32 Size) const {
    u64 Consumed = ConsumedMask | (IsWrite ? WriteMask : 0) |
                   (static_cast<u64>(EncodeSize(Size)) << AddressBits) |
                   (PC & AddressMask);
    return __scoped_atomic_compare_exchange_n(
        Watchpoint, &Encoded, Consumed, false, __ATOMIC_RELAXED,
        __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }

  bool Consume(u64 *Watchpoint, void *&Peer, int &PeerAccess,
               u32 &PeerSize) const {
    u64 Old = __scoped_atomic_exchange_n(
        Watchpoint, ConsumedMask, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    Peer = reinterpret_cast<void *>(Old & AddressMask);
    PeerAccess = (Old & WriteMask) ? CSAN_ACCESS_WRITE : 0;
    PeerSize = (Old & SizeMask) >> AddressBits;
    return !(Old & ConsumedMask);
  }

  void Remove(u64 *Watchpoint) const {
    __scoped_atomic_store_n(Watchpoint, Invalid, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
  }
};

} // namespace __csan

#ifdef CSAN_DEFINED_SCOPED_ATOMICS
#undef __scoped_atomic_load_n
#undef __scoped_atomic_store_n
#undef __scoped_atomic_compare_exchange_n
#undef __scoped_atomic_exchange_n
#undef CSAN_DEFINED_SCOPED_ATOMICS
#endif

#endif // CSAN_WATCH_H
