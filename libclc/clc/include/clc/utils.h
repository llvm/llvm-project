//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_UTILS_H__
#define __CLC_UTILS_H__

#define __CLC_CONCAT(x, y) x##y
#define __CLC_XCONCAT(x, y) __CLC_CONCAT(x, y)

#define __CLC_STR(x) #x
#define __CLC_XSTR(x) __CLC_STR(x)

#endif // __CLC_UTILS_H__
