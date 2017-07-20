//===-- scudo_utils.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_utils.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_UTILS_H_
#define SCUDO_UTILS_H_

#include <string.h>

#include "sanitizer_common/sanitizer_common.h"

namespace __scudo {

template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  static_assert(sizeof(Dest) == sizeof(Source), "Sizes are not equal!");
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

void NORETURN dieWithMessage(const char *Format, ...);

enum CPUFeature {
  CRC32CPUFeature = 0,
  MaxCPUFeature,
};
bool testCPUFeature(CPUFeature feature);

INLINE u64 rotl(const u64 X, int K) {
  return (X << K) | (X >> (64 - K));
}

// XoRoShiRo128+ PRNG (http://xoroshiro.di.unimi.it/).
struct XoRoShiRo128Plus {
 public:
  void init() {
    if (UNLIKELY(!GetRandom(reinterpret_cast<void *>(State), sizeof(State)))) {
      // Early processes (eg: init) do not have /dev/urandom yet, but we still
      // have to provide them with some degree of entropy. Not having a secure
      // seed is not as problematic for them, as they are less likely to be
      // the target of heap based vulnerabilities exploitation attempts.
      State[0] = NanoTime();
      State[1] = 0;
    }
    fillCache();
  }
  u8 getU8() {
    if (UNLIKELY(isCacheEmpty()))
      fillCache();
    const u8 Result = static_cast<u8>(CachedBytes & 0xff);
    CachedBytes >>= 8;
    CachedBytesAvailable--;
    return Result;
  }
  u64 getU64() { return next(); }

 private:
  u8 CachedBytesAvailable;
  u64 CachedBytes;
  u64 State[2];
  u64 next() {
    const u64 S0 = State[0];
    u64 S1 = State[1];
    const u64 Result = S0 + S1;
    S1 ^= S0;
    State[0] = rotl(S0, 55) ^ S1 ^ (S1 << 14);
    State[1] = rotl(S1, 36);
    return Result;
  }
  bool isCacheEmpty() {
    return CachedBytesAvailable == 0;
  }
  void fillCache() {
    CachedBytes = next();
    CachedBytesAvailable = sizeof(CachedBytes);
  }
};

typedef XoRoShiRo128Plus ScudoPrng;

}  // namespace __scudo

#endif  // SCUDO_UTILS_H_
