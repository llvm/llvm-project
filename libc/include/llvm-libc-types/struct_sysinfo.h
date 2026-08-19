//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct sysinfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_SYSINFO_H
#define LLVM_LIBC_TYPES_STRUCT_SYSINFO_H

// Kernel struct defined in the UAPI headers. Include it instead of defining it
// ourselves.

#include <linux/sysinfo.h>

#endif // LLVM_LIBC_TYPES_STRUCT_SYSINFO_H
