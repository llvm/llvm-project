//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header file for getpwent function and internal helpers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_GETPWENT_H
#define LLVM_LIBC_SRC_PWD_GETPWENT_H

#include "hdr/types/struct_passwd.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Overrides the default password file path for testing purposes.
void set_passwd_path(const char *path);

} // namespace internal

// Internal helper function to open or rewind the password file.
ErrorOr<int> setpwent_impl();

// Internal helper function to close the password file.
ErrorOr<int> endpwent_impl();

// Reads the next entry from the password database.
struct passwd *getpwent();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_GETPWENT_H
