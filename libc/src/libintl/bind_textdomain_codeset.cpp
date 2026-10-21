//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of bind_textdomain_codeset (stub).
///
//===----------------------------------------------------------------------===//

#include "src/libintl/bind_textdomain_codeset.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, bind_textdomain_codeset,
                   (const char *domainname, const char *codeset)) {
  // TODO: Implement message catalogs. For now, validate domainname and return
  // the requested codeset (or nullptr when querying).
  if (domainname == nullptr || domainname[0] == '\0' || codeset == nullptr)
    return nullptr;
  return const_cast<char *>(codeset);
}

} // namespace LIBC_NAMESPACE_DECL
