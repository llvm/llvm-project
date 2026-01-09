//===-- asan_aix.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// AIX-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_AIX
#  include "asan_mapping.h"
#  include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

void TryReExecWithoutASLR() {
  // Allowed to fail and do nothing.
}

void AsanCheckIncompatibleRT() {}

void AsanCheckDynamicRTPrereqs() {}

void InitializePlatformExceptionHandlers() {}

void* AsanDoesNotSupportStaticLinkage() { return 0; }

void InitializePlatformInterceptors() {}
void AsanApplyToGlobals(globals_op_fptr op, const void* needle) {}

uptr FindDynamicShadowStart() {
  UNREACHABLE("AIX does not use dynamic shadow offset!");
  return 0;
}

void FlushUnneededASanShadowMemory(uptr p, uptr size) {
  ReleaseMemoryPagesToOS(MemToShadow(p), MemToShadow(p + size));
}

}  // namespace __asan

#endif  // SANITIZER_AIX
