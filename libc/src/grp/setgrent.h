//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header file for setgrent function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_GRP_SETGRENT_H
#define LLVM_LIBC_SRC_GRP_SETGRENT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Rewinds the group database stream to the beginning.
void setgrent();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_GRP_SETGRENT_H
