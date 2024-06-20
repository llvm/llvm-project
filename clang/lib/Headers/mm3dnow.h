/*===---- mm3dnow.h - 3DNow! intrinsics ------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

// 3dNow intrinsics are no longer supported, and this header remains only as a
// stub for users who were including it to get to _m_prefetch or
// _m_prefetchw. Such uses should prefer x86intrin.h.

#ifndef _MM3DNOW_H_INCLUDED
#define _MM3DNOW_H_INCLUDED

#include <mmintrin.h>
#include <prfchwintrin.h>

#undef __DEFAULT_FN_ATTRS

#endif
