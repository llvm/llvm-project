//===-- Implementation of fputc for baremetal -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fputc.h"

#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/fputc_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fputc, (int c, ::FILE *stream)) {
  return fputc_internal(c, stream);
}

} // namespace LIBC_NAMESPACE_DECL
