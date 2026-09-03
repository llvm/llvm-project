//===-- Standard C header <type_guarding.h> --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LLVM_LIBC_TYPE_GUARDING_H
#define _LLVM_LIBC_TYPE_GUARDING_H

#include "__llvm-libc-common.h"
#include "llvm-libc-macros/float16-macro.h"
#include "llvm-libc-macros/size_t-macro.h"
#include "llvm-libc-types/myType.h"
#include <stdint.h>

#ifdef LIBC_TYPES_HAS_FLOAT16
#include "llvm-libc-types/float16.h"
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_SIZE_T
#include "llvm-libc-types/size_t.h"
#include "llvm-libc-types/ssize_t.h"
#endif // LIBC_TYPES_HAS_SIZE_T

#endif // _LLVM_LIBC_TYPE_GUARDING_H
