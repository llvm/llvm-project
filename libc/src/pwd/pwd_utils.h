//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of helper functions for pwd.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_PWD_UTILS_H
#define LLVM_LIBC_SRC_PWD_PWD_UTILS_H

#include "hdr/types/struct_passwd.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Parses a colon-separated password database line into a struct passwd.
ErrorOr<struct passwd> parse_passwd_line(char *line);

} // namespace internal

namespace passwd {

// Overrides the default password file path for testing purposes.
void TESTONLY_set_passwd_path(const char *path);

// Opens or rewinds the password file.
ErrorOr<void> open();

// Closes the password file.
ErrorOr<void> close();

// Reads the next entry from the password database.
ErrorOr<struct passwd *> read_next();

} // namespace passwd
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_PWD_UTILS_H
