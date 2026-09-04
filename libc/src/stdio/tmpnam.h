//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of tmpnam, a POSIX function that generate a string that is a
/// valid pathname that does not name an existing file.
/// See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/tmpnam.html
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_TMPNAM_H
#define LLVM_LIBC_SRC_STDIO_TMPNAM_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *tmpnam(char *s);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_TMPNAM_H
