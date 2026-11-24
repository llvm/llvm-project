//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of openlog.
///
//===----------------------------------------------------------------------===//

#include "src/syslog/openlog.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/syslog/linux/syslog_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, openlog,
                   (const char *ident, int option, int facility)) {
  syslog_internal::open_log(ident, option, facility);
}

} // namespace LIBC_NAMESPACE_DECL
