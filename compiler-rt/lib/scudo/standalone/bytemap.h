//===-- bytemap.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_BYTEMAP_H_
#define SCUDO_BYTEMAP_H_

#include "atomic_helpers.h"
#include "common.h"
#include "mutex.h"

namespace scudo {

template <uptr Size> class FlatByteMap {
public:
  void initLinkerInitialized() {
    Map = reinterpret_cast<u8 *>(map(nullptr, Size, "scudo:bytemap"));
  }
  void init() { initLinkerInitialized(); }

  void unmapTestOnly() { unmap(reinterpret_cast<void *>(Map), Size); }

  void set(uptr Index, u8 Value) {
    DCHECK_LT(Index, Size);
    DCHECK_EQ(0U, Map[Index]);
    Map[Index] = Value;
  }
  u8 operator[](uptr Index) {
    DCHECK_LT(Index, Size);
    return Map[Index];
  }

private:
  u8 *Map;
};

template <uptr Level1Size, uptr Level2Size> class TwoLevelByteMap {
public:
  void initLinkerInitialized() {
    Level1Map = reinterpret_cast<atomic_uptr *>(
        map(nullptr, sizeof(atomic_uptr) * Level1Size, "scudo:bytemap"));
  }
  void init() {
    initLinkerInitialized();
    Mutex.init();
  }

  void reset() {
    for (uptr I = 0; I < Level1Size; I++) {
      u8 *P = get(I);
      if (!P)
        continue;
      unmap(P, Level2Size);
    }
    memset(Level1Map, 0, sizeof(atomic_uptr) * Level1Size);
  }

  void unmapTestOnly() {
    reset();
    unmap(reinterpret_cast<void *>(Level1Map),
          sizeof(atomic_uptr) * Level1Size);
  }

  uptr size() const { return Level1Size * Level2Size; }

  void set(uptr Index, u8 Value) {
    DCHECK_LT(Index, Level1Size * Level2Size);
    u8 *Level2Map = getOrCreate(Index / Level2Size);
    DCHECK_EQ(0U, Level2Map[Index % Level2Size]);
    Level2Map[Index % Level2Size] = Value;
  }

  u8 operator[](uptr Index) const {
    DCHECK_LT(Index, Level1Size * Level2Size);
    u8 *Level2Map = get(Index / Level2Size);
    if (!Level2Map)
      return 0;
    return Level2Map[Index % Level2Size];
  }

private:
  u8 *get(uptr Index) const {
    DCHECK_LT(Index, Level1Size);
    return reinterpret_cast<u8 *>(
        atomic_load(&Level1Map[Index], memory_order_acquire));
  }

  u8 *getOrCreate(uptr Index) {
    u8 *Res = get(Index);
    if (!Res) {
      SpinMutexLock L(&Mutex);
      if (!(Res = get(Index))) {
        Res = reinterpret_cast<u8 *>(map(nullptr, Level2Size, "scudo:bytemap"));
        atomic_store(&Level1Map[Index], reinterpret_cast<uptr>(Res),
                     memory_order_release);
      }
    }
    return Res;
  }

  atomic_uptr *Level1Map;
  StaticSpinMutex Mutex;
};

} // namespace scudo

#endif // SCUDO_BYTEMAP_H_
