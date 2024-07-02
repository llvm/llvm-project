
//===-- Definition of type locale_t ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_LOCALE_T_H
#define LLVM_LIBC_TYPES_LOCALE_T_H

// HACK(@izaakschroeder): Placeholder.
// NOTE: According to `libcxx` the `locale_t` type has to be at least
// coercible to a `bool`.
typedef void* locale_t;

#endif // LLVM_LIBC_TYPES_LOCALE_T_H
