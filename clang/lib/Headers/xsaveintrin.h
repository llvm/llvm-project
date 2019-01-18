/*===---- xsaveintrin.h - XSAVE intrinsic ----------------------------------===
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

#ifndef __IMMINTRIN_H
#error "Never use <xsaveintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __XSAVEINTRIN_H
#define __XSAVEINTRIN_H

#ifdef _MSC_VER
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("xsave")))

static __inline__ void __DEFAULT_FN_ATTRS
_xsave(void *__p, unsigned long long __m) {
  __builtin_ia32_xsave(__p, __m);
}

static __inline__ void __DEFAULT_FN_ATTRS
_xrstor(void *__p, unsigned long long __m) {
  __builtin_ia32_xrstor(__p, __m);
}

#ifndef _MSC_VER
#define _xgetbv(A) __builtin_ia32_xgetbv((long long)(A))
#define _xsetbv(A, B) __builtin_ia32_xsetbv((unsigned int)(A), (unsigned long long)(B));
#else
#ifdef __cplusplus
extern "C" {
#endif
unsigned __int64 __cdecl _xgetbv(unsigned int);
void __cdecl _xsetbv(unsigned int, unsigned __int64);
#ifdef __cplusplus
}
#endif
#endif /* _MSC_VER */

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS
_xsave64(void *__p, unsigned long long __m) {
  __builtin_ia32_xsave64(__p, __m);
}

static __inline__ void __DEFAULT_FN_ATTRS
_xrstor64(void *__p, unsigned long long __m) {
  __builtin_ia32_xrstor64(__p, __m);
}

#endif

#undef __DEFAULT_FN_ATTRS

#endif
