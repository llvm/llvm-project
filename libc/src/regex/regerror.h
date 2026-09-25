//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for regerror.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_REGEX_REGERROR_H
#define LLVM_LIBC_SRC_REGEX_REGERROR_H

#include "hdr/types/regex_t.h"
#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"

namespace LIBC_NAMESPACE_DECL {

size_t regerror(int errcode, const regex_t *__restrict preg,
                char *__restrict errbuf, size_t errbuf_size);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_REGEX_REGERROR_H
