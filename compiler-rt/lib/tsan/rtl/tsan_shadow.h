//===-- tsan_shadow.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TSAN_SHADOW_H
#define TSAN_SHADOW_H

#include "tsan_defs.h"
#include "tsan_trace.h"

namespace __tsan {

// FastState (from most significant bit):
//   ignore          : 1
//   tid             : kTidBits
//   unused          : -
//   history_size    : 3
//   epoch           : kClkBits
class FastState {
 public:
  FastState(u64 tid, u64 epoch) {
    x_ = tid << kTidShift;
    x_ |= epoch;
    DCHECK_EQ(tid, this->tid());
    DCHECK_EQ(epoch, this->epoch());
    DCHECK_EQ(GetIgnoreBit(), false);
  }

  explicit FastState(u64 x) : x_(x) {}

  u64 raw() const { return x_; }

  u64 tid() const {
    u64 res = (x_ & ~kIgnoreBit) >> kTidShift;
    return res;
  }

  u64 TidWithIgnore() const {
    u64 res = x_ >> kTidShift;
    return res;
  }

  u64 epoch() const {
    u64 res = x_ & ((1ull << kClkBits) - 1);
    return res;
  }

  void IncrementEpoch() {
    u64 old_epoch = epoch();
    x_ += 1;
    DCHECK_EQ(old_epoch + 1, epoch());
    (void)old_epoch;
  }

  void SetIgnoreBit() { x_ |= kIgnoreBit; }
  void ClearIgnoreBit() { x_ &= ~kIgnoreBit; }
  bool GetIgnoreBit() const { return (s64)x_ < 0; }

  void SetHistorySize(int hs) {
    CHECK_GE(hs, 0);
    CHECK_LE(hs, 7);
    x_ = (x_ & ~(kHistoryMask << kHistoryShift)) | (u64(hs) << kHistoryShift);
  }

  ALWAYS_INLINE
  int GetHistorySize() const {
    return (int)((x_ >> kHistoryShift) & kHistoryMask);
  }

  void ClearHistorySize() { SetHistorySize(0); }

  ALWAYS_INLINE
  u64 GetTracePos() const {
    const int hs = GetHistorySize();
    // When hs == 0, the trace consists of 2 parts.
    const u64 mask = (1ull << (kTracePartSizeBits + hs + 1)) - 1;
    return epoch() & mask;
  }

 private:
  friend class Shadow;
  static const int kTidShift = 64 - kTidBits - 1;
  static const u64 kIgnoreBit = 1ull << 63;
  static const u64 kFreedBit = 1ull << 63;
  static const u64 kHistoryShift = kClkBits;
  static const u64 kHistoryMask = 7;
  u64 x_;
};

// Shadow (from most significant bit):
//   freed           : 1
//   tid             : kTidBits
//   is_atomic       : 1
//   is_read         : 1
//   size_log        : 2
//   addr0           : 3
//   epoch           : kClkBits
class Shadow : public FastState {
 public:
  explicit Shadow(u64 x) : FastState(x) {}

  explicit Shadow(const FastState &s) : FastState(s.x_) { ClearHistorySize(); }

  void SetAddr0AndSizeLog(u64 addr0, unsigned kAccessSizeLog) {
    DCHECK_EQ((x_ >> kClkBits) & 31, 0);
    DCHECK_LE(addr0, 7);
    DCHECK_LE(kAccessSizeLog, 3);
    x_ |= ((kAccessSizeLog << 3) | addr0) << kClkBits;
    DCHECK_EQ(kAccessSizeLog, size_log());
    DCHECK_EQ(addr0, this->addr0());
  }

  void SetWrite(unsigned kAccessIsWrite) {
    DCHECK_EQ(x_ & kReadBit, 0);
    if (!kAccessIsWrite)
      x_ |= kReadBit;
    DCHECK_EQ(kAccessIsWrite, IsWrite());
  }

  void SetAtomic(bool kIsAtomic) {
    DCHECK(!IsAtomic());
    if (kIsAtomic)
      x_ |= kAtomicBit;
    DCHECK_EQ(IsAtomic(), kIsAtomic);
  }

  bool IsAtomic() const { return x_ & kAtomicBit; }

  bool IsZero() const { return x_ == 0; }

  static inline bool TidsAreEqual(const Shadow s1, const Shadow s2) {
    u64 shifted_xor = (s1.x_ ^ s2.x_) >> kTidShift;
    DCHECK_EQ(shifted_xor == 0, s1.TidWithIgnore() == s2.TidWithIgnore());
    return shifted_xor == 0;
  }

  static ALWAYS_INLINE bool Addr0AndSizeAreEqual(const Shadow s1,
                                                 const Shadow s2) {
    u64 masked_xor = ((s1.x_ ^ s2.x_) >> kClkBits) & 31;
    return masked_xor == 0;
  }

  static ALWAYS_INLINE bool TwoRangesIntersect(Shadow s1, Shadow s2,
                                               unsigned kS2AccessSize) {
    bool res = false;
    u64 diff = s1.addr0() - s2.addr0();
    if ((s64)diff < 0) {  // s1.addr0 < s2.addr0
      // if (s1.addr0() + size1) > s2.addr0()) return true;
      if (s1.size() > -diff)
        res = true;
    } else {
      // if (s2.addr0() + kS2AccessSize > s1.addr0()) return true;
      if (kS2AccessSize > diff)
        res = true;
    }
    DCHECK_EQ(res, TwoRangesIntersectSlow(s1, s2));
    DCHECK_EQ(res, TwoRangesIntersectSlow(s2, s1));
    return res;
  }

  u64 ALWAYS_INLINE addr0() const { return (x_ >> kClkBits) & 7; }
  u64 ALWAYS_INLINE size() const { return 1ull << size_log(); }
  bool ALWAYS_INLINE IsWrite() const { return !IsRead(); }
  bool ALWAYS_INLINE IsRead() const { return x_ & kReadBit; }

  // The idea behind the freed bit is as follows.
  // When the memory is freed (or otherwise unaccessible) we write to the shadow
  // values with tid/epoch related to the free and the freed bit set.
  // During memory accesses processing the freed bit is considered
  // as msb of tid. So any access races with shadow with freed bit set
  // (it is as if write from a thread with which we never synchronized before).
  // This allows us to detect accesses to freed memory w/o additional
  // overheads in memory access processing and at the same time restore
  // tid/epoch of free.
  void MarkAsFreed() { x_ |= kFreedBit; }

  bool IsFreed() const { return x_ & kFreedBit; }

  bool GetFreedAndReset() {
    bool res = x_ & kFreedBit;
    x_ &= ~kFreedBit;
    return res;
  }

  bool ALWAYS_INLINE IsBothReadsOrAtomic(bool kIsWrite, bool kIsAtomic) const {
    bool v = x_ & ((u64(kIsWrite ^ 1) << kReadShift) |
                   (u64(kIsAtomic) << kAtomicShift));
    DCHECK_EQ(v, (!IsWrite() && !kIsWrite) || (IsAtomic() && kIsAtomic));
    return v;
  }

  bool ALWAYS_INLINE IsRWNotWeaker(bool kIsWrite, bool kIsAtomic) const {
    bool v = ((x_ >> kReadShift) & 3) <= u64((kIsWrite ^ 1) | (kIsAtomic << 1));
    DCHECK_EQ(v, (IsAtomic() < kIsAtomic) ||
                     (IsAtomic() == kIsAtomic && !IsWrite() <= !kIsWrite));
    return v;
  }

  bool ALWAYS_INLINE IsRWWeakerOrEqual(bool kIsWrite, bool kIsAtomic) const {
    bool v = ((x_ >> kReadShift) & 3) >= u64((kIsWrite ^ 1) | (kIsAtomic << 1));
    DCHECK_EQ(v, (IsAtomic() > kIsAtomic) ||
                     (IsAtomic() == kIsAtomic && !IsWrite() >= !kIsWrite));
    return v;
  }

 private:
  static const u64 kReadShift = 5 + kClkBits;
  static const u64 kReadBit = 1ull << kReadShift;
  static const u64 kAtomicShift = 6 + kClkBits;
  static const u64 kAtomicBit = 1ull << kAtomicShift;

  u64 size_log() const { return (x_ >> (3 + kClkBits)) & 3; }

  static bool TwoRangesIntersectSlow(const Shadow s1, const Shadow s2) {
    if (s1.addr0() == s2.addr0())
      return true;
    if (s1.addr0() < s2.addr0() && s1.addr0() + s1.size() > s2.addr0())
      return true;
    if (s2.addr0() < s1.addr0() && s2.addr0() + s2.size() > s1.addr0())
      return true;
    return false;
  }
};

const RawShadow kShadowRodata = (RawShadow)-1;  // .rodata shadow marker

}  // namespace __tsan

#endif
