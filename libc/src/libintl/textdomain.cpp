//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of textdomain (stub).
///
//===----------------------------------------------------------------------===//

#include "src/libintl/textdomain.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, textdomain, (const char *domainname)) {
  // TODO: Implement message catalogs. For now, return the default "messages"
  // domain when querying or resetting, and domainname otherwise.
  if (domainname == nullptr || domainname[0] == '\0')
    return const_cast<char *>("messages");
  return const_cast<char *>(domainname);
}

} // namespace LIBC_NAMESPACE_DECL
