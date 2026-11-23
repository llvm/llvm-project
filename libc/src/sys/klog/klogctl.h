//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for klogctl.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_KLOG_KLOGCTL_H
#define LLVM_LIBC_SRC_SYS_KLOG_KLOGCTL_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int klogctl(int type, char *bufp, int len);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_KLOG_KLOGCTL_H
