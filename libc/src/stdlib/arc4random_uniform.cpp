//===-- Implementation of arc4random_uniform --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random_uniform.h"

namespace LIBC_NAMESPACE_DECL {

uint32_t arc4random_uniform(uint32_t upper_bound) {
  // TODO: Implement arc4random_uniform
  (void)upper_bound;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
