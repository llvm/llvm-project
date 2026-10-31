//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for dcngettext.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LIBINTL_DCNGETTEXT_H
#define LLVM_LIBC_SRC_LIBINTL_DCNGETTEXT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *dcngettext(const char *domainname, const char *msgid1, const char *msgid2,
                 unsigned long int n, int category);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LIBINTL_DCNGETTEXT_H
