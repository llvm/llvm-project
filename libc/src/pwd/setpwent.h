//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of setpwent.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_SETPWENT_H
#define LLVM_LIBC_SRC_PWD_SETPWENT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void setpwent();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_SETPWENT_H
