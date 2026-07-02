//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of getpwent.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_GETPWENT_H
#define LLVM_LIBC_SRC_PWD_GETPWENT_H

#include "hdr/types/struct_passwd.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

struct passwd *getpwent();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_GETPWENT_H
