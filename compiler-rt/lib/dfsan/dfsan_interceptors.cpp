//===-- dfsan_interceptors.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// Interceptors for standard library functions.
//===----------------------------------------------------------------------===//

#include <sys/syscall.h>
#include <unistd.h>

#include "dfsan/dfsan.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;

namespace {

bool interceptors_initialized;

}  // namespace

INTERCEPTOR(void *, mmap, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF_T offset) {
  void *res;

  // interceptors_initialized is set to true during preinit_array, when we're
  // single-threaded.  So we don't need to worry about accessing it atomically.
  if (!interceptors_initialized)
    res = (void *)syscall(__NR_mmap, addr, length, prot, flags, fd, offset);
  else
    res = REAL(mmap)(addr, length, prot, flags, fd, offset);

  if (res != (void *)-1)
    dfsan_set_label(0, res, RoundUpTo(length, GetPageSizeCached()));
  return res;
}

INTERCEPTOR(void *, mmap64, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF64_T offset) {
  void *res = REAL(mmap64)(addr, length, prot, flags, fd, offset);
  if (res != (void *)-1)
    dfsan_set_label(0, res, RoundUpTo(length, GetPageSizeCached()));
  return res;
}

INTERCEPTOR(int, munmap, void *addr, SIZE_T length) {
  int res = REAL(munmap)(addr, length);
  if (res != -1)
    dfsan_set_label(0, addr, RoundUpTo(length, GetPageSizeCached()));
  return res;
}

namespace __dfsan {
void InitializeInterceptors() {
  CHECK(!interceptors_initialized);

  INTERCEPT_FUNCTION(mmap);
  INTERCEPT_FUNCTION(mmap64);
  INTERCEPT_FUNCTION(munmap);

  interceptors_initialized = true;
}
}  // namespace __dfsan
