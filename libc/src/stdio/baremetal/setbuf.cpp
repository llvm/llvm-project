//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the bare-metal implementation of setbuf.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/setbuf.h"
#include "hdr/stdio_macros.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, setbuf,
                   (::FILE *__restrict stream, char *__restrict buffer)) {
  __llvm_libc_stdio_set_buffer(stream, buffer, BUFSIZ,
                               buffer == nullptr ? _IONBF : _IOFBF);
}

} // namespace LIBC_NAMESPACE_DECL
