//===-- sanitizer_flat_map.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_FLAT_MAP_H
#define SANITIZER_FLAT_MAP_H

#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_local_address_space_view.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

// Call these callbacks on mmap/munmap.
struct NoOpMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnUnmap(uptr p, uptr size) const {}
};

// Maps integers in rage [0, kSize) to u8 values.
template <u64 kSize, typename AddressSpaceViewTy = LocalAddressSpaceView>
class FlatMap {
 public:
  using AddressSpaceView = AddressSpaceViewTy;
  void Init() { internal_memset(map_, 0, sizeof(map_)); }

  constexpr uptr size() const { return kSize; }

  void set(uptr idx, u8 val) {
    CHECK_LT(idx, kSize);
    CHECK_EQ(0U, map_[idx]);
    map_[idx] = val;
  }
  u8 operator[](uptr idx) {
    CHECK_LT(idx, kSize);
    // FIXME: CHECK may be too expensive here.
    return map_[idx];
  }

 private:
  u8 map_[kSize];
};

// TwoLevelByteMap maps integers in range [0, kSize1*kSize2) to u8 values.
// It is implemented as a two-dimensional array: array of kSize1 pointers
// to kSize2-byte arrays. The secondary arrays are mmaped on demand.
// Each value is initially zero and can be set to something else only once.
// Setting and getting values from multiple threads is safe w/o extra locking.
template <u64 kSize1, u64 kSize2,
          typename AddressSpaceViewTy = LocalAddressSpaceView,
          class MapUnmapCallback = NoOpMapUnmapCallback>
class TwoLevelMap {
 public:
  using AddressSpaceView = AddressSpaceViewTy;
  void Init() {
    internal_memset(map1_, 0, sizeof(map1_));
    mu_.Init();
  }

  void TestOnlyUnmap() {
    for (uptr i = 0; i < kSize1; i++) {
      u8 *p = Get(i);
      if (!p)
        continue;
      MapUnmapCallback().OnUnmap(reinterpret_cast<uptr>(p), kSize2);
      UnmapOrDie(p, kSize2);
    }
  }

  constexpr uptr size() const { return kSize1 * kSize2; }
  constexpr uptr size1() const { return kSize1; }
  constexpr uptr size2() const { return kSize2; }

  void set(uptr idx, u8 val) {
    CHECK_LT(idx, kSize1 * kSize2);
    u8 *map2 = GetOrCreate(idx / kSize2);
    CHECK_EQ(0U, map2[idx % kSize2]);
    map2[idx % kSize2] = val;
  }

  u8 operator[](uptr idx) const {
    CHECK_LT(idx, kSize1 * kSize2);
    u8 *map2 = Get(idx / kSize2);
    if (!map2)
      return 0;
    auto value_ptr = AddressSpaceView::Load(&map2[idx % kSize2]);
    return *value_ptr;
  }

 private:
  u8 *Get(uptr idx) const {
    CHECK_LT(idx, kSize1);
    return reinterpret_cast<u8 *>(
        atomic_load(&map1_[idx], memory_order_acquire));
  }

  u8 *GetOrCreate(uptr idx) {
    u8 *res = Get(idx);
    if (!res) {
      SpinMutexLock l(&mu_);
      if (!(res = Get(idx))) {
        res = (u8 *)MmapOrDie(kSize2, "TwoLevelMap");
        MapUnmapCallback().OnMap(reinterpret_cast<uptr>(res), kSize2);
        atomic_store(&map1_[idx], reinterpret_cast<uptr>(res),
                     memory_order_release);
      }
    }
    return res;
  }

  atomic_uintptr_t map1_[kSize1];
  StaticSpinMutex mu_;
};

template <u64 kSize, typename AddressSpaceViewTy = LocalAddressSpaceView>
using FlatByteMap = FlatMap<kSize, AddressSpaceViewTy>;

template <u64 kSize1, u64 kSize2,
          typename AddressSpaceViewTy = LocalAddressSpaceView,
          class MapUnmapCallback = NoOpMapUnmapCallback>
using TwoLevelByteMap =
    TwoLevelMap<kSize1, kSize2, AddressSpaceViewTy, MapUnmapCallback>;

}  // namespace __sanitizer

#endif
