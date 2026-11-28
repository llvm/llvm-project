//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for closelog.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYSLOG_CLOSELOG_H
#define LLVM_LIBC_SRC_SYSLOG_CLOSELOG_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void closelog();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYSLOG_CLOSELOG_H
