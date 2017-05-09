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

// Tiny PRNG based on https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
// The state (128 bits) will be stored in thread local storage.
struct Xorshift128Plus {
 public:
  void initFromURandom();
  u64 getNext() {
    u64 x = State[0];
    const u64 y = State[1];
    State[0] = y;
    x ^= x << 23;
    State[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return State[1] + y;
  }
 private:
  u64 State[2];
};

}  // namespace __scudo

#endif  // SCUDO_UTILS_H_
