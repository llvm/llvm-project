//===-- Implementation of funlockfile -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/funlockfile.h"
#include "src/__support/File/file.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, funlockfile, (::FILE * stream)) {
  reinterpret_cast<__llvm_libc::File *>(stream)->unlock();
}

} // namespace __llvm_libc
