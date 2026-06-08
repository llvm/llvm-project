//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for internal error reporting.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERR_REPORT_H
#define LLVM_LIBC_SRC_ERR_REPORT_H

#include "src/__support/arg_list.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace err_reporting {

void report(bool show_err, int err_num, const char *fmt,
            internal::ArgList &args);

} // namespace err_reporting
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_ERR_REPORT_H
