//===-- Internal implementation of ungetc for baremetal --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/baremetal/ungetc_internal.h"

#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

int ungetc_internal(int c, ::FILE *stream) {
  return store_ungetc_value(stream, c);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
