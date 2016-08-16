//===-- asan_descriptions.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_descriptions.cc.
// TODO(filcab): Most struct definitions should move to the interface headers.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"

namespace __asan {

enum ShadowKind : u8 {
  kShadowKindLow,
  kShadowKindGap,
  kShadowKindHigh,
};
static const char *const ShadowNames[] = {"low shadow", "shadow gap",
                                          "high shadow"};

struct ShadowAddressDescription {
  uptr addr;
  ShadowKind kind;
  u8 shadow_byte;
};

bool GetShadowAddressInformation(uptr addr, ShadowAddressDescription *descr);
bool DescribeAddressIfShadow(uptr addr);

}  // namespace __asan
