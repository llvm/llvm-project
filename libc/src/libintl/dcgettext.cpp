//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of dcgettext (stub).
///
//===----------------------------------------------------------------------===//

#include "src/libintl/dcgettext.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, dcgettext,
                   ([[maybe_unused]] const char *domainname, const char *msgid,
                    [[maybe_unused]] int category)) {
  // TODO: Implement message catalogs. For now, return msgid unchanged.
  return const_cast<char *>(msgid);
}

} // namespace LIBC_NAMESPACE_DECL
