//===-- Implementation of fileno
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fileno.h"
#include "src/__support/File/file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, fileno, (::FILE * stream)) {
  int result = get_fileno(reinterpret_cast<LIBC_NAMESPACE::File *>(stream));
  return result;
}

} // namespace LIBC_NAMESPACE
