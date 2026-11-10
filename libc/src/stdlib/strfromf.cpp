//===-- Implementation of strfromf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfromf.h"
#include "hdr/types/size_t.h"
#include "src/stdlib/str_from_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, strfromf,
                   (char *__restrict s, size_t n, const char *__restrict format,
                    float fp)) {
  return internal::strfromfloat_impl(s, n, format, fp);
}

} // namespace LIBC_NAMESPACE_DECL
