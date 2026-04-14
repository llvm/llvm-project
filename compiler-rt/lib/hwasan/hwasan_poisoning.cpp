//===-- hwasan_poisoning.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//

#include "hwasan_poisoning.h"

#include "hwasan.h"
#include "hwasan_mapping.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_linux.h"

namespace __hwasan {

uptr TagMemory(uptr p, uptr size, tag_t tag) {
  uptr start = RoundDownTo(p, kShadowAlignment);
  uptr end = RoundUpTo(p + size, kShadowAlignment);
  return TagMemoryAligned(start, end - start, tag);
}

}  // namespace __hwasan

// --- Implementation of LSan-specific functions --- {{{1
namespace __lsan {
bool WordIsPoisoned(uptr addr) {
  if (!InTaggableRegion(addr))
    return false;
  tag_t Tag = *reinterpret_cast<tag_t*>(__hwasan::MemToShadow(addr));
  return Tag >= (1U << __hwasan::HwasanTagBits());
}
}  // namespace __lsan
