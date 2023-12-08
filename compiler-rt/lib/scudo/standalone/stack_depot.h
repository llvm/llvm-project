//===-- stack_depot.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_STACK_DEPOT_H_
#define SCUDO_STACK_DEPOT_H_

#include "atomic_helpers.h"
#include "mutex.h"

namespace scudo {

class MurMur2HashBuilder {
  static const u32 M = 0x5bd1e995;
  static const u32 Seed = 0x9747b28c;
  static const u32 R = 24;
  u32 H;

public:
  explicit MurMur2HashBuilder(u32 Init = 0) { H = Seed ^ Init; }
  void add(u32 K) {
    K *= M;
    K ^= K >> R;
    K *= M;
    H *= M;
    H ^= K;
  }
  u32 get() {
    u32 X = H;
    X ^= X >> 13;
    X *= M;
    X ^= X >> 15;
    return X;
  }
};

class StackDepot {
  HybridMutex RingEndMu;
  u32 RingEnd = 0;

  // This data structure stores a stack trace for each allocation and
  // deallocation when stack trace recording is enabled, that may be looked up
  // using a hash of the stack trace. The lower bits of the hash are an index
  // into the Tab array, which stores an index into the Ring array where the
  // stack traces are stored. As the name implies, Ring is a ring buffer, so a
  // stack trace may wrap around to the start of the array.
  //
  // Each stack trace in Ring is prefixed by a stack trace marker consisting of
  // a fixed 1 bit in bit 0 (this allows disambiguation between stack frames
  // and stack trace markers in the case where instruction pointers are 4-byte
  // aligned, as they are on arm64), the stack trace hash in bits 1-32, and the
  // size of the stack trace in bits 33-63.
  //
  // The insert() function is potentially racy in its accesses to the Tab and
  // Ring arrays, but find() is resilient to races in the sense that, barring
  // hash collisions, it will either return the correct stack trace or no stack
  // trace at all, even if two instances of insert() raced with one another.
  // This is achieved by re-checking the hash of the stack trace before
  // returning the trace.

#if SCUDO_SMALL_STACK_DEPOT
  static const uptr TabBits = 4;
#else
  static const uptr TabBits = 16;
#endif
  static const uptr TabSize = 1 << TabBits;
  static const uptr TabMask = TabSize - 1;
  atomic_u32 Tab[TabSize] = {};

#if SCUDO_SMALL_STACK_DEPOT
  static const uptr RingBits = 4;
#else
  static const uptr RingBits = 19;
#endif
  static const uptr RingSize = 1 << RingBits;
  static const uptr RingMask = RingSize - 1;
  atomic_u64 Ring[RingSize] = {};

public:
  // Insert hash of the stack trace [Begin, End) into the stack depot, and
  // return the hash.
  u32 insert(uptr *Begin, uptr *End) {
    MurMur2HashBuilder B;
    for (uptr *I = Begin; I != End; ++I)
      B.add(u32(*I) >> 2);
    u32 Hash = B.get();

    u32 Pos = Hash & TabMask;
    u32 RingPos = atomic_load_relaxed(&Tab[Pos]);
    u64 Entry = atomic_load_relaxed(&Ring[RingPos]);
    u64 Id = (u64(End - Begin) << 33) | (u64(Hash) << 1) | 1;
    if (Entry == Id)
      return Hash;

    ScopedLock Lock(RingEndMu);
    RingPos = RingEnd;
    atomic_store_relaxed(&Tab[Pos], RingPos);
    atomic_store_relaxed(&Ring[RingPos], Id);
    for (uptr *I = Begin; I != End; ++I) {
      RingPos = (RingPos + 1) & RingMask;
      atomic_store_relaxed(&Ring[RingPos], *I);
    }
    RingEnd = (RingPos + 1) & RingMask;
    return Hash;
  }

  // Look up a stack trace by hash. Returns true if successful. The trace may be
  // accessed via operator[] passing indexes between *RingPosPtr and
  // *RingPosPtr + *SizePtr.
  bool find(u32 Hash, uptr *RingPosPtr, uptr *SizePtr) const {
    u32 Pos = Hash & TabMask;
    u32 RingPos = atomic_load_relaxed(&Tab[Pos]);
    if (RingPos >= RingSize)
      return false;
    u64 Entry = atomic_load_relaxed(&Ring[RingPos]);
    u64 HashWithTagBit = (u64(Hash) << 1) | 1;
    if ((Entry & 0x1ffffffff) != HashWithTagBit)
      return false;
    u32 Size = u32(Entry >> 33);
    if (Size >= RingSize)
      return false;
    *RingPosPtr = (RingPos + 1) & RingMask;
    *SizePtr = Size;
    MurMur2HashBuilder B;
    for (uptr I = 0; I != Size; ++I) {
      RingPos = (RingPos + 1) & RingMask;
      B.add(u32(atomic_load_relaxed(&Ring[RingPos])) >> 2);
    }
    return B.get() == Hash;
  }

  u64 operator[](uptr RingPos) const {
    return atomic_load_relaxed(&Ring[RingPos & RingMask]);
  }
};

} // namespace scudo

#endif // SCUDO_STACK_DEPOT_H_
