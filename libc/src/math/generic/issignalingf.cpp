//===-- Implementation of issignalingf function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/issignalingf.h"
#include "src/__support/math/issignalingf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, issignalingf, (float x)) {
  return math::issignalingf(x);
}

} // namespace LIBC_NAMESPACE_DECL
