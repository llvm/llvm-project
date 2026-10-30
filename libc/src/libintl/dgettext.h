//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for dgettext.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LIBINTL_DGETTEXT_H
#define LLVM_LIBC_SRC_LIBINTL_DGETTEXT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *dgettext(const char *domainname, const char *msgid);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LIBINTL_DGETTEXT_H
