//===-- Implementation of arc4random ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random.h"
#include "src/__support/CPP/optional.h"
#include "src/stdlib/linux/vdso_rng.h"

namespace LIBC_NAMESPACE_DECL {

uint32_t arc4random() {
  using namespace vdso_rng;
  uint32_t result = 0;
  if (cpp::optional<LocalState::Guard> guard = local_state.get())
    guard->fill(&result, sizeof(result));
  else
    fallback_rng_fill(&result, sizeof(result));
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
