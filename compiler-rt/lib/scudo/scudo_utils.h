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

enum  CPUFeature {
  SSE4_2 = 0,
  ENUM_CPUFEATURE_MAX
};
bool testCPUFeature(CPUFeature feature);

// Tiny PRNG based on https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
// The state (128 bits) will be stored in thread local storage.
struct Xorshift128Plus {
 public:
  Xorshift128Plus();
  u64 Next() {
    u64 x = State_0_;
    const u64 y = State_1_;
    State_0_ = y;
    x ^= x << 23;
    State_1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
    return State_1_ + y;
  }
 private:
  u64 State_0_;
  u64 State_1_;
};

} // namespace __scudo

#endif  // SCUDO_UTILS_H_
