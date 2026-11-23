//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines types to be used with (CLC) vstore and vload functions. These are
// vector types whose alignment is that of their respective scalar types.
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

typedef __CLC_GENTYPE __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE);

#else

typedef __CLC_GENTYPE __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)
    __attribute__((aligned(sizeof(__CLC_SCALAR_GENTYPE))));

#endif
