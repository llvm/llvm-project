//===-- Implementation of fflush for baremetal -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

// Baremetal uses unbuffered I/O, so there is nothing to flush.
LLVM_LIBC_FUNCTION(int, fflush, (::FILE * stream )) {
    (void) stream;
    // TODO: Shall we have an embedding API for fflush?
    return 0;
}

} // namespace LIBC_NAMESPACE_DECL
