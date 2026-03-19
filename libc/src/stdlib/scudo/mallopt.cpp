//===-- Scudo mallopt -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/mallopt.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mallopt, (int param, int value)) {
  if (param == M_DECAY_TIME) {
    if (SCUDO_ANDROID) {
      Allocator.releaseToOS(scudo::ReleaseToOS::Force);
      CHECK(value >= -1 && value <= 1);
      if (value == 1)
        value = INT32_MAX;
    }

    Allocator.setOption(scudo::Option::ReleaseInterval,
                        static_cast<scudo::sptr>(value));
    return 1;
  }
  if (param == M_PURGE) {
    Allocator.releaseToOS(scudo::ReleaseToOS::Force);
    return 1;
  }
  if (param == M_PURGE_FAST) {
    Allocator.releaseToOS(scudo::ReleaseToOS::ForceFast);
    return 1;
  }
  if (param == M_PURGE_ALL) {
    Allocator.releaseToOS(scudo::ReleaseToOS::ForceAll);
    return 1;
  }
  if (param == M_LOG_STATS) {
    Allocator.printStats();
    Allocator.printFragmentationInfo();
    return 1;
  }

  scudo::Option Option;
  switch (param) {
  case M_MEMTAG_TUNING:
    Option = scudo::Option::MemtagTuning;
    break;
  case M_THREAD_DISABLE_MEM_INIT:
    Option = scudo::Option::ThreadDisableMemInit;
    break;
  case M_CACHE_COUNT_MAX:
    Option = scudo::Option::MaxCacheEntriesCount;
    break;
  case M_CACHE_SIZE_MAX:
    Option = scudo::Option::MaxCacheEntrySize;
    break;
  case M_TSDS_COUNT_MAX:
    Option = scudo::Option::MaxTSDsCount;
    break;
  default:
    return 0;
  }

  return Allocator.setOption(Option, static_cast<scudo::sptr>(value));
}

} // namespace LIBC_NAMESPACE_DECL
