//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of mkdtemp, a POSIX function that creates a unique temporary
/// directory from a template string ending in at least six 'X' characters.
///
/// Replaces the trailing X's with random characters from the POSIX portable
/// filename character set, creates the directory with 0700 permissions,
/// and returns the pathname, retrying automatically on name collision. See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_MKDTEMP_H
#define LLVM_LIBC_SRC_STDLIB_MKDTEMP_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Create a unique temporary directory from a template string.
///
/// \param tmpl Template string ending in at least six 'X' characters.
/// \return Pointer to the modified template string on success, nullptr on
/// error.
char *mkdtemp(char *tmpl);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_MKDTEMP_H
