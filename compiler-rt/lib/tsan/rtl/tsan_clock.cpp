//===-- tsan_clock.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_clock.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {

#if defined(__SSE3__)
const uptr kClockSize128 = kMaxSid * sizeof(Epoch) / sizeof(m128);
#endif

VectorClock::VectorClock() {
  Reset();
}

void VectorClock::Reset() {
#if defined(__SSE3__)
  m128 z = _mm_setzero_si128();
  m128* clk = reinterpret_cast<m128*>(clk_);
  for (uptr i = 0; i < kClockSize128; i++)
    _mm_store_si128(&clk[i], z);
#else
  for (uptr i = 0; i < kMaxSid; i++)
    clk_[i] = kEpochZero;
#endif
}

void VectorClock::Acquire(const VectorClock* src) {
  if (!src)
    return;
#if defined(__SSE3__)
  m128* __restrict vdst = reinterpret_cast<m128*>(clk_);
  m128 const* __restrict vsrc = reinterpret_cast<m128 const*>(src->clk_);
  for (uptr i = 0; i < kClockSize128; i++) {
    m128 s = _mm_load_si128(&vsrc[i]);
    m128 d = _mm_load_si128(&vdst[i]);
    m128 m = _mm_max_epu16(s, d);
    _mm_store_si128(&vdst[i], m);
  }
#else
  for (uptr i = 0; i < kMaxSid; i++)
    clk_[i] = max(clk_[i], src->clk_[i]);
#endif
}

NOINLINE
VectorClock* AllocClock(VectorClock** dstp) {
  if (!*dstp)
    *dstp = New<VectorClock>();
  return *dstp;
}

void VectorClock::Release(VectorClock** dstp) const {
  VectorClock* dst = AllocClock(dstp);
  dst->Acquire(this);
}

void VectorClock::ReleaseStore(VectorClock** dstp) const {
  VectorClock* dst = AllocClock(dstp);
  *dst = *this;
}

void VectorClock::operator=(const VectorClock& other) {
#if defined(__SSE3__)
  m128* __restrict vdst = reinterpret_cast<m128*>(clk_);
  m128 const* __restrict vsrc = reinterpret_cast<m128 const*>(other.clk_);
  for (uptr i = 0; i < kClockSize128; i++) {
    m128 s = _mm_load_si128(&vsrc[i]);
    _mm_store_si128(&vdst[i], s);
  }
#else
  for (uptr i = 0; i < kMaxSid; i++)
    clk_[i] = other.clk_[i];
#endif
}

void VectorClock::ReleaseStoreAcquire(VectorClock** dstp) {
  VectorClock* __restrict dst = AllocClock(dstp);
  for (uptr i = 0; i < kMaxSid; i++) {
    Epoch tmp = dst->clk_[i];
    dst->clk_[i] = clk_[i];
    clk_[i] = max(clk_[i], tmp);
  }
}

void VectorClock::ReleaseAcquire(VectorClock** dstp) {
  VectorClock* __restrict other = AllocClock(dstp);
#if defined(__SSE3__)
  m128* dst = reinterpret_cast<m128*>(other->clk_);
  m128* clk = reinterpret_cast<m128*>(clk_);
  for (uptr i = 0; i < kClockSize128; i++) {
    m128 c = _mm_load_si128(&clk[i]);
    m128 d = _mm_load_si128(&dst[i]);
    m128 m = _mm_max_epu16(c, d);
    _mm_store_si128(&dst[i], m);
    _mm_store_si128(&clk[i], m);
  }
#else
  for (uptr i = 0; i < kMaxSid; i++) {
    dst->clk_[i] = max(dst->clk_[i], clk_[i]);
    clk_[i] = dst->clk_[i];
  }
#endif
}

}  // namespace __tsan
