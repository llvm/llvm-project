/*===---- stddef.h - Basic type definitions --------------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __STDDEF_H
#define __STDDEF_H

typedef __typeof__(((int*)0)-((int*)0)) ptrdiff_t;
#ifndef _SIZE_T
#define _SIZE_T
typedef __typeof__(sizeof(int)) size_t;
#endif
#ifndef __cplusplus
#ifndef _WCHAR_T
#define _WCHAR_T
typedef __WCHAR_TYPE__ wchar_t;
#endif
#endif

#undef NULL
#ifdef __cplusplus
#undef __null  // VC++ hack.
#define NULL __null
#else
#define NULL ((void*)0)
#endif

// Some C libraries expect to see a wint_t here. Others (notably MinGW) will use
// __WINT_TYPE__ directly; accomodate both by requiring __need_wint_t
#if defined(__need_wint_t) && !defined(_WINT_T)
#define _WINT_T
typedef __WINT_TYPE__ wint_t;
#endif

#define offsetof(t, d) __builtin_offsetof(t, d)

#endif /* __STDDEF_H */
