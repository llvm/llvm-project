/*===---- __clang_openmp_runtime_wrapper.h - CUDA runtime support ----------===
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

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 *
 */

#ifndef __CLANG_OPENMP_RUNTIME_WRAPPER_H__
#define __CLANG_OPENMP_RUNTIME_WRAPPER_H__

//  Is this a device pass
#if defined(__AMDGCN__) || defined(__NVPTX__)

#define __forceinline__ __attribute__((always_inline))

// Add any macro definitions here that prevent standard header
// files from doing any function definitions that would not be
// valid for device pass.

// __NO_INLINE__ prevents some x86 optimized definitions
#define __NO_INLINE__ 1

#endif // is a device pass

#endif // __CLANG_OPENMP_RUNTIME_WRAPPER_H__
