//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for bind_textdomain_codeset.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LIBINTL_BIND_TEXTDOMAIN_CODESET_H
#define LLVM_LIBC_SRC_LIBINTL_BIND_TEXTDOMAIN_CODESET_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *bind_textdomain_codeset(const char *domainname, const char *codeset);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LIBINTL_BIND_TEXTDOMAIN_CODESET_H
