/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 *******************************************************************************/
extern const __constant bool __oclc_finite_only_opt;
extern const __constant bool __oclc_unsafe_math_opt;
extern const __constant bool __oclc_daz_opt;
extern const __constant bool __oclc_correctly_rounded_sqrt32;
extern const __constant bool __oclc_wavefrontsize64;
extern const __constant int __oclc_ISA_version;
extern const __constant int __oclc_ABI_version;

void kernel device_libs(__global float *status) {

 if (__oclc_finite_only_opt)            status[0] = 1.0;
 if (__oclc_unsafe_math_opt)            status[1] = 1.0;
 if (__oclc_daz_opt)                    status[2] = 1.0;
 if (__oclc_correctly_rounded_sqrt32)   status[3] = 1.0;
 if (__oclc_wavefrontsize64)            status[4] = 1.0;
 if (__oclc_ISA_version)                status[5] = 1.0;
 if (__oclc_ABI_version)                status[6] = 1.0;

 // Math functions to test AMDGPULibCalls Folding optimizations
 // fold_sincos()
 float x = 0.25;
 status[7] = sin(x) + cos(x);
 status[8] = cos(x) + sin(x);

 // fold_rootn()
 float y = 725.0;
 status[9] = rootn(y, 3);
 status[10] = rootn(y, -1);
 status[11] = rootn(y, -2);

 // fold_pow()
 float z = 12.16;
 status[12] = pow(z, (float) 0.5);
 status[13] = powr(y, (float) 7.23);
}
