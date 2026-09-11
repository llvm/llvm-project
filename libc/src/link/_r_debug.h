//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of _r_debug.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LINK__R_DEBUG_H
#define LLVM_LIBC_SRC_LINK__R_DEBUG_H

#include "hdr/types/struct_r_debug.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern struct r_debug _r_debug;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LINK__R_DEBUG_H
