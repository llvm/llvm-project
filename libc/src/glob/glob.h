//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for glob.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_GLOB_GLOB_H
#define LLVM_LIBC_SRC_GLOB_GLOB_H

#include "hdr/types/glob_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int glob(const char *__restrict pattern, int flags,
         int (*errfunc)(const char *epath, int eerrno),
         glob_t *__restrict pglob);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_GLOB_GLOB_H
