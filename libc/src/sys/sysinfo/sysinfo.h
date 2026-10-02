//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for sysinfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SYSINFO_SYSINFO_H
#define LLVM_LIBC_SRC_SYS_SYSINFO_SYSINFO_H

#include "hdr/types/struct_sysinfo.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int sysinfo(struct sysinfo *info);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SYSINFO_SYSINFO_H
