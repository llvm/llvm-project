//===-- hwasan_poisoning.cc -------------------------------------*- C++ -*-===//
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

#include "hwasan_mapping.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_linux.h"

namespace __hwasan {

uptr TagMemoryAligned(uptr p, uptr size, tag_t tag) {
  CHECK(IsAligned(p, kShadowAlignment));
  CHECK(IsAligned(size, kShadowAlignment));
  uptr shadow_start = MemToShadow(p);
  uptr shadow_size = MemToShadowSize(size);

  uptr page_size = GetPageSizeCached();
  uptr page_start = RoundUpTo(shadow_start, page_size);
  uptr page_end = RoundDownTo(shadow_start + shadow_size, page_size);
  uptr threshold = common_flags()->clear_shadow_mmap_threshold;
  if (SANITIZER_LINUX &&
      UNLIKELY(page_end >= page_start + threshold && tag == 0)) {
    internal_memset((void *)shadow_start, tag, page_start - shadow_start);
    internal_memset((void *)page_end, tag,
                    shadow_start + shadow_size - page_end);
    // For an anonymous private mapping MADV_DONTNEED will return a zero page on
    // Linux.
    ReleaseMemoryPagesToOSAndZeroFill(page_start, page_end);
  } else {
    internal_memset((void *)shadow_start, tag, shadow_size);
  }
  return AddTagToPointer(p, tag);
}

uptr TagMemory(uptr p, uptr size, tag_t tag) {
  uptr start = RoundDownTo(p, kShadowAlignment);
  uptr end = RoundUpTo(p + size, kShadowAlignment);
  return TagMemoryAligned(start, end - start, tag);
}

}  // namespace __hwasan
