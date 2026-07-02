//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of mkstemp, a POSIX function that creates a unique temporary
/// file from a template string ending in at least six 'X' characters.
///
/// Replaces the trailing X's with random characters from the POSIX portable
/// filename character set, opens the file exclusively, and returns an open
/// file descriptor, retrying automatically on name collision. See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_MKSTEMP_H
#define LLVM_LIBC_SRC_STDLIB_MKSTEMP_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int mkstemp(char *tmpl);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_MKSTEMP_H
