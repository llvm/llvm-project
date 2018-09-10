/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
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
 */

#if __CLC_FPSIZE == 64
#define __CLC_NAN __builtin_nan("")
#define ZERO 0.0
#elif __CLC_FPSIZE == 32
#define __CLC_NAN NAN
#define ZERO 0.0f
#elif __CLC_FPSIZE == 16
#define __CLC_NAN (half)NAN
#define ZERO 0.0h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sqrt(__CLC_GENTYPE val) {
  return val < ZERO ? __CLC_NAN : __clc_llvm_intr_sqrt(val);
}

#undef __CLC_NAN
#undef ZERO
