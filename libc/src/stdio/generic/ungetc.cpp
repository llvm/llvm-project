//===-- Implementation of ungetc ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/ungetc.h"
#include "src/__support/File/file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, ungetc, (int c, ::FILE *stream)) {
  return reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->ungetc(c);
}

} // namespace LIBC_NAMESPACE
