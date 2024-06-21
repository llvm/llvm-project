//===-- Implementation of tls for arm -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h" // TLSDescriptor

namespace LIBC_NAMESPACE {


void init_tls(TLSDescriptor &) {
  // TODO: implement me! https://github.com/llvm/llvm-project/issues/96326
}

void cleanup_tls(uintptr_t, uintptr_t) {
  // TODO: implement me! https://github.com/llvm/llvm-project/issues/96326
}

} // namespace LIBC_NAMESPACE
