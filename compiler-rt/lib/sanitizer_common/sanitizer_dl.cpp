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

}  // namespace __sanitizer
