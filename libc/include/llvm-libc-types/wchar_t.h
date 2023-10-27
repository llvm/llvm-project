//===-- Definition of wchar_t types ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_WCHAR_T_H__
#define __LLVM_LIBC_TYPES_WCHAR_T_H__

// Since __need_wchar_t is defined, we get the definition of wchar_t from the
// standalone C header stddef.h. Also, because __need_wchar_t is defined,
// including stddef.h will pull only the type wchar_t and nothing else.
#define __need_wchar_t
#include <stddef.h>
#undef __need_wchar_t

#endif // __LLVM_LIBC_TYPES_WCHAR_T_H__
