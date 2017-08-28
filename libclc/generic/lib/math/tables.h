/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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

#define TABLE_SPACE __constant

#define TABLE_MANGLE(NAME) __clc_##NAME

#define DECLARE_TABLE(TYPE,NAME,LENGTH) \
    TABLE_SPACE TYPE NAME [ LENGTH ]

#define TABLE_FUNCTION(TYPE,TABLE,NAME) \
    TYPE TABLE_MANGLE(NAME)(size_t idx) { \
        return TABLE[idx]; \
    }

#define TABLE_FUNCTION_DECL(TYPE, NAME) \
    TYPE TABLE_MANGLE(NAME)(size_t idx);

#define USE_TABLE(NAME, IDX) \
    TABLE_MANGLE(NAME)(IDX)

TABLE_FUNCTION_DECL(float2, loge_tbl);
TABLE_FUNCTION_DECL(float, log_inv_tbl);
TABLE_FUNCTION_DECL(float2, log2_tbl);
TABLE_FUNCTION_DECL(uint4,  pibits_tbl);
TABLE_FUNCTION_DECL(float2, sinhcosh_tbl);
TABLE_FUNCTION_DECL(float2, cbrt_tbl);
TABLE_FUNCTION_DECL(float, exp_tbl);
TABLE_FUNCTION_DECL(float2, exp_tbl_ep);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

TABLE_FUNCTION_DECL(double2, ln_tbl);
TABLE_FUNCTION_DECL(double2, atan_jby256_tbl);
TABLE_FUNCTION_DECL(double2, two_to_jby64_ep_tbl);
TABLE_FUNCTION_DECL(double2, sinh_tbl);
TABLE_FUNCTION_DECL(double2, cosh_tbl);
TABLE_FUNCTION_DECL(double, cbrt_inv_tbl);
TABLE_FUNCTION_DECL(double2, cbrt_dbl_tbl);
TABLE_FUNCTION_DECL(double2, cbrt_rem_tbl);

#endif // cl_khr_fp64
