//===-- Implementation of arc4random_buf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random_buf.h"
#include "src/__support/CPP/optional.h"
#include "src/stdlib/linux/vdso_rng.h"

namespace LIBC_NAMESPACE_DECL {

void arc4random_buf(void *buf, size_t len) {
  using namespace vdso_rng;
  if (cpp::optional<LocalState::Guard> guard = local_state.get())
    guard->fill(buf, len);
  else
    fallback_rng_fill(buf, len);
}

} // namespace LIBC_NAMESPACE_DECL
