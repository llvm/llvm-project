//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of dngettext (stub).
///
//===----------------------------------------------------------------------===//

#include "src/libintl/dngettext.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, dngettext,
                   ([[maybe_unused]] const char *domainname, const char *msgid1,
                    const char *msgid2, unsigned long int n)) {
  // TODO: Implement message catalogs. For now, return singular or plural based
  // on n == 1.
  return const_cast<char *>(n == 1 ? msgid1 : msgid2);
}

} // namespace LIBC_NAMESPACE_DECL
