//===-- tsan_update_shadow_word_inl.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Body of the hottest inner loop.
// If we wrap this body into a function, compilers (both gcc and clang)
// produce sligtly less efficient code.
//===----------------------------------------------------------------------===//
do {
  StatInc(thr, StatShadowProcessed);
  const unsigned kAccessSize = 1 << kAccessSizeLog;
  u64 *sp = &shadow_mem[idx];
  old = LoadShadow(sp);
  if (old.IsZero()) {
    StatInc(thr, StatShadowZero);
    if (store_word)
      StoreIfNotYetStored(sp, &store_word);
    // The above StoreIfNotYetStored could be done unconditionally
    // and it even shows 4% gain on synthetic benchmarks (r4307).
    break;
  }
  // is the memory access equal to the previous?
  if (Shadow::Addr0AndSizeAreEqual(cur, old)) {
    StatInc(thr, StatShadowSameSize);
    // same thread?
    if (Shadow::TidsAreEqual(old, cur)) {
      StatInc(thr, StatShadowSameThread);
      if (old.IsRWWeakerOrEqual(kAccessIsWrite, kIsAtomic))
        StoreIfNotYetStored(sp, &store_word);
      break;
    }
    StatInc(thr, StatShadowAnotherThread);
    if (HappensBefore(old, thr)) {
      if (old.IsRWWeakerOrEqual(kAccessIsWrite, kIsAtomic))
        StoreIfNotYetStored(sp, &store_word);
      break;
    }
    if (old.IsBothReadsOrAtomic(kAccessIsWrite, kIsAtomic))
      break;
    goto RACE;
  }
  // Do the memory access intersect?
  if (Shadow::TwoRangesIntersect(old, cur, kAccessSize)) {
    StatInc(thr, StatShadowIntersect);
    if (Shadow::TidsAreEqual(old, cur)) {
      StatInc(thr, StatShadowSameThread);
      break;
    }
    StatInc(thr, StatShadowAnotherThread);
    if (old.IsBothReadsOrAtomic(kAccessIsWrite, kIsAtomic))
      break;
    if (HappensBefore(old, thr))
      break;
    goto RACE;
  }
  // The accesses do not intersect.
  StatInc(thr, StatShadowNotIntersect);
  break;
} while (0);
