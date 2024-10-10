//===-- Definition of macros to be used with complex functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_COMPLEX_MACROS_H
#define __LLVM_LIBC_MACROS_COMPLEX_MACROS_H

#ifndef __STDC_NO_COMPLEX__

#define __STDC_VERSION_COMPLEX_H__ 202311L

#define complex _Complex
#define _Complex_I ((float _Complex)1.0fi)

#ifdef _Imaginary
#define imaginary _Imaginary
#define _Imaginary_I ((float _Imaginary)1.0i)

#define I _Imaginary_I
#else
#define I _Complex_I
#endif

#endif

#endif // __LLVM_LIBC_MACROS_COMPLEX_MACROS_H
