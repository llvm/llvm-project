//===-- Implementation of arc4random_buf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random_buf.h"
#include "src/stdlib/linux/vsdo_rng.h"

namespace LIBC_NAMESPACE_DECL {

void arc4random_buf(void *buf, size_t len) {
  vdso_rng::LocalState::Guard guard = vdso_rng::get();
}

} // namespace LIBC_NAMESPACE_DECL
