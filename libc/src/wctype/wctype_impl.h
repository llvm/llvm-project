//===-- Implementation header for iswctype ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCTYPE_WCTYPE_H
#define LLVM_LIBC_SRC_WCTYPE_WCTYPE_H

#include "hdr/types/wctype_t.h"
#include "hdr/types/wint_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

wctype_t wctype(const char *property);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCTYPE_WCTYPE_H
