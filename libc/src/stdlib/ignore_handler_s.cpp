//===-- Implementation header for ignore_handler_s --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/ignore_handler_s.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, ignore_handler_s,
                   (const char *__restrict, void *__restrict, errno_t)) {}

} // namespace LIBC_NAMESPACE_DECL
