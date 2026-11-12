//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of bindtextdomain (stub).
///
//===----------------------------------------------------------------------===//

#include "src/libintl/bindtextdomain.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, bindtextdomain,
                   (const char *domainname, const char *dirname)) {
  // TODO: Implement message catalogs. For now, validate domainname and return
  // the default locale directory when querying or dirname when setting.
  if (domainname == nullptr || domainname[0] == '\0')
    return nullptr;
  if (dirname == nullptr)
    return const_cast<char *>("/usr/share/locale");
  return const_cast<char *>(dirname);
}

} // namespace LIBC_NAMESPACE_DECL
