//===-- Implementation header for twalk_r -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEARCH_TWALK_R_H
#define LLVM_LIBC_SRC_SEARCH_TWALK_R_H

#include "hdr/types/VISIT.h"
#include "hdr/types/posix_tnode.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
void twalk_r(const __llvm_libc_tnode *root,
             void (*action)(const __llvm_libc_tnode *, VISIT, void *),
             void *closure);
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEARCH_TWALK_R_H
