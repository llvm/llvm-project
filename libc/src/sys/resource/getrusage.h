//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of getrusage.
///
//===----------------------------------------------------------------------===//

#ifndef LVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H_
#define LVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H_

#include "hdr/types/struct_rusage.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getrusage(int who, struct rusage *usage);

} // namespace LIBC_NAMESPACE_DECL

#endif  // LLVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H_
