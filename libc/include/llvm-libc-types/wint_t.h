//===-- Definition of wint_t types ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_WINT_T_H
#define LLVM_LIBC_TYPES_WINT_T_H

// Since __need_wint_t is defined, we get the definition of wint_t from the
// standalone C header stddef.h. Also, because __need_wint_t is defined,
// including stddef.h will pull only the type wint_t and nothing else.
#define __need_wint_t
#include <stddef.h>
#undef __need_wint_t

#endif // LLVM_LIBC_TYPES_WINT_T_H
