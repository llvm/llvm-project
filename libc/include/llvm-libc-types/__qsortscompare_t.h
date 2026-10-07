//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Type definition for __qsortscompare_t.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES___QSORTSCOMPARE_T_H
#define LLVM_LIBC_TYPES___QSORTSCOMPARE_T_H

typedef int (*__qsortscompare_t)(const void *, const void *, void *);

#endif // LLVM_LIBC_TYPES___QSORTSCOMPARE_T_H
