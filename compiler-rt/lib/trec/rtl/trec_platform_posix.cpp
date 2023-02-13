//===-- trec_platform_posix.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
// POSIX-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "trec_platform.h"
#include "trec_rtl.h"

namespace __trec {

static const char kShadowMemoryMappingWarning[] =
    "FATAL: %s can not madvise shadow region [%zx, %zx] with %s (errno: %d)\n";
static const char kShadowMemoryMappingHint[] =
    "HINT: if %s is not supported in your environment, you may set "
    "TREC_OPTIONS=%s=0\n";

static void DontDumpShadow(uptr addr, uptr size) {
  if (common_flags()->use_madv_dontdump)
    if (!DontDumpShadowMemory(addr, size)) {
      Printf(kShadowMemoryMappingWarning, SanitizerToolName, addr, addr + size,
             "MADV_DONTDUMP", errno);
      Printf(kShadowMemoryMappingHint, "MADV_DONTDUMP", "use_madv_dontdump");
      Die();
    }
}

#if !SANITIZER_GO

static void ProtectRange(uptr beg, uptr end) {
  CHECK_LE(beg, end);
  if (beg == end)
    return;
  if (beg != (uptr)MmapFixedNoAccess(beg, end - beg)) {
    Printf("FATAL: TraceRecorder can not protect [%zx,%zx]\n", beg, end);
    Printf("FATAL: Make sure you are not using unlimited stack\n");
    Die();
  }
}

#endif

}  // namespace __trec

#endif  // SANITIZER_POSIX
