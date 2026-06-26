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
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

bool parse_passwd_line(char *line, struct passwd *pwd);

} // namespace internal

void setpwent_impl();
void endpwent_impl();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_PWD_UTILS_H
