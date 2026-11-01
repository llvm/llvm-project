//===-- sanitizer_dl.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file has helper functions that depend on libc's dynamic loading
// introspection.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_dl.h"

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_GLIBC
#  include <dlfcn.h>
#endif

namespace __sanitizer {
extern const char *SanitizerToolName;

const char *DladdrSelfFName(void) {
  // go-tsan can't link libdl before it was merged into glibc 2.34.
#if SANITIZER_GLIBC && !SANITIZER_GO
  Dl_info info;
  int ret = dladdr((void *)&SanitizerToolName, &info);
  if (ret) {
    return info.dli_fname;
  }
#endif

  return nullptr;
}

char* DladdrElfHeaderBase(void* ld, char* addr) {
  // go-tsan can't link libdl before it was merged into glibc 2.34.
#if SANITIZER_GLIBC && !SANITIZER_GO
  Dl_info info;
  if (dladdr(ld, &info) && info.dli_fbase)
    addr = (char*)info.dli_fbase;
#endif  // SANITIZER_GLIBC
  return addr;
}

void ClearDlerror() {
#if SANITIZER_GLIBC && !SANITIZER_GO
  // Starting with glibc 2.34
  // (https://sourceware.org/git/?p=glibc.git;h=fada9018199c), failed dlfcn
  // calls store a per-thread error struct in TLS (__libc_dlerror_result)
  // instead of a pthread_key_t, and free it in __libc_thread_freeres() after
  // pthread TSD destructors have already unregistered the thread from the
  // sanitizer. Clear dlerror() while the thread is still tracked so that LSan
  // does not report the pending error buffer as a leak if a leak check runs
  // before __libc_thread_freeres().
  dlerror();
  dlerror();
#endif
}

}  // namespace __sanitizer
