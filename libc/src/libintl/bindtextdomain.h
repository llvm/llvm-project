//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for bindtextdomain.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LIBINTL_BINDTEXTDOMAIN_H
#define LLVM_LIBC_SRC_LIBINTL_BINDTEXTDOMAIN_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *bindtextdomain(const char *domainname, const char *dirname);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LIBINTL_BINDTEXTDOMAIN_H
