//===-- Implementation of ungetc for baremetal -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/ungetc.h"

#include "src/__support/common.h"
#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ungetc, (int c, ::FILE *stream)) {
  return internal::ungetc_internal(c, stream);
}

} // namespace LIBC_NAMESPACE_DECL
