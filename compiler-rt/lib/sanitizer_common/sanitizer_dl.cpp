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

#include <dlfcn.h>

#include "sanitizer_common/sanitizer_platform.h"

namespace __sanitizer {
extern const char *SanitizerToolName;

int dladdr_self_fname(const char **fname) {
#if SANITIZER_GLIBC
  Dl_info info;
  int ret = dladdr((void *)&SanitizerToolName, &info);
  *fname = info.dli_fname;
  return ret;
#else
  *fname = nullptr;
  return 0;
#endif
}

}  // namespace __sanitizer
