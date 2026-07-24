//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of macros from sys/cdefs.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_SYS_CDEFS_MACROS_H
#define LLVM_LIBC_MACROS_SYS_CDEFS_MACROS_H

#ifdef __cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

#ifndef __THROW
#ifdef __cplusplus
#define __THROW throw()
#else
#define __THROW
#endif
#endif

#endif // LLVM_LIBC_MACROS_SYS_CDEFS_MACROS_H
