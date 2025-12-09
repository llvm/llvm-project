//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_INTEGER_DEFINITIONS_H__
#define __CLC_INTEGER_DEFINITIONS_H__

#define CHAR_BIT 8
#define INT_MAX 2147483647
#ifndef INT_MIN
#define INT_MIN (-2147483647 - 1)
#endif
#define LONG_MAX 0x7fffffffffffffffL
#ifndef LONG_MIN
#define LONG_MIN (-0x7fffffffffffffffL - 1)
#endif
#define CHAR_MAX SCHAR_MAX
#define CHAR_MIN SCHAR_MIN
#define SCHAR_MAX 127
#ifndef SCHAR_MIN
#define SCHAR_MIN (-127 - 1)
#endif
#define SHRT_MAX 32767
#ifndef SHRT_MIN
#define SHRT_MIN (-32767 - 1)
#endif
#define UCHAR_MAX 255
#define UCHAR_MIN 0
#define USHRT_MAX 65535
#define USHRT_MIN 0
#define UINT_MAX 0xffffffff
#define UINT_MIN 0
#define ULONG_MAX 0xffffffffffffffffUL
#define ULONG_MIN 0UL

#endif // __CLC_INTEGER_DEFINITIONS_H__
