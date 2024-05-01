//===-- Proxy for struct flock64  -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_HDR_TYPES_STRUCT_FLOCK64_H
#define LLVM_LIBC_HDR_TYPES_STRUCT_FLOCK64_H

#ifdef LIBC_FULL_BUILD

#include "include/llvm-libc-types/struct_flock64.h"

#else

#include <fcntl.h>

#endif // LIBC_FULL_BUILD

#endif // LLVM_LIBC_HDR_TYPES_STRUCT_FLOCK64_H
