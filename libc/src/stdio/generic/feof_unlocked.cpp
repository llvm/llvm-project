//===-- Implementation of feof_unlocked -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/feof_unlocked.h"
#include "src/__support/File/file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, feof_unlocked, (::FILE * stream)) {
  return reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->iseof_unlocked();
}

} // namespace LIBC_NAMESPACE
