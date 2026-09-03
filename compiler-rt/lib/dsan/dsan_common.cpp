//=-- dsan_common.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Implementation of common double-free checking functionality.
//
//===----------------------------------------------------------------------===//

#include "dsan_common.h"

#include "sanitizer_common/sanitizer_common.h"

namespace __dsan {

void InitCommonDsan() {
  // DoubleFreeSanitizer doesn't need complex initialization.
  // Detection is done inline in RegisterDeallocation.
}

}  // namespace __dsan

extern "C" {
SANITIZER_INTERFACE_WEAK_DEF(const char*, __dsan_default_options, void) {
  return "";
}
}  // extern "C"
