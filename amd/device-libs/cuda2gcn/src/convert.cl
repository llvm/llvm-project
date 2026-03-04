/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((const))

#define CONVERTM(A,B,m,n) ATTR B __nv_##A##2##B##_##m(A x) \
    { return convert_##B##_##n(x); }

#define CONVERT(A,B) \
    CONVERTM(A, B, rd, rtn) \
    CONVERTM(A, B, rn, rte) \
    CONVERTM(A, B, ru, rtp) \
    CONVERTM(A, B, rz, rtz)

//-------- T __nv_double2float_rd
//-------- T __nv_double2float_rn
//-------- T __nv_double2float_ru
//-------- T __nv_double2float_rz
CONVERT(double, float)

//-------- T __nv_double2int_rd
//-------- T __nv_double2int_rn
//-------- T __nv_double2int_ru
//-------- T __nv_double2int_rz
CONVERT(double, int)

//-------- T __nv_float2int_rd
//-------- T __nv_float2int_rn
//-------- T __nv_float2int_ru
//-------- T __nv_float2int_rz
CONVERT(float, int)

//-------- T __nv_int2float_rd
//-------- T __nv_int2float_rn
//-------- T __nv_int2float_ru
//-------- T __nv_int2float_rz
CONVERT(int, float)

//-------- T __nv_double2uint_rd
//-------- T __nv_double2uint_rn
//-------- T __nv_double2uint_ru
//-------- T __nv_double2uint_rz
CONVERT(double, uint)

//-------- T __nv_float2uint_rd
//-------- T __nv_float2uint_rn
//-------- T __nv_float2uint_ru
//-------- T __nv_float2uint_rz
CONVERT(float, uint)

//-------- T __nv_uint2double_rd
//-------- T __nv_uint2double_rn
//-------- T __nv_uint2double_ru
//-------- T __nv_uint2double_rz
CONVERT(uint, double)

//-------- T __nv_uint2float_rd
//-------- T __nv_uint2float_rn
//-------- T __nv_uint2float_ru
//-------- T __nv_uint2float_rz
CONVERT(uint, float)

#define CONVERT2LLM(A,B,m,n) ATTR long __nv_##A##2ll_##m(A x) \
    { return convert_long_##n(x); }

#define CONVERT2LL(A) \
    CONVERT2LLM(A, long, rd, rtn) \
    CONVERT2LLM(A, long, rn, rte) \
    CONVERT2LLM(A, long, ru, rtp) \
    CONVERT2LLM(A, long, rz, rtz)

//-------- T __nv_double2ll_rd
//-------- T __nv_double2ll_rn
//-------- T __nv_double2ll_ru
//-------- T __nv_double2ll_rz
CONVERT2LL(double)

//-------- T __nv_float2ll_rd
//-------- T __nv_float2ll_rn
//-------- T __nv_float2ll_ru
//-------- T __nv_float2ll_rz
CONVERT2LL(float)

#define CONVERT2ULLM(A,B,m,n) ATTR ulong __nv_##A##2ull_##m(A x) \
    { return convert_ulong_##n(x); }

#define CONVERT2ULL(A) \
    CONVERT2ULLM(A, ulong, rd, rtn) \
    CONVERT2ULLM(A, ulong, rn, rte) \
    CONVERT2ULLM(A, ulong, ru, rtp) \
    CONVERT2ULLM(A, ulong, rz, rtz)

//-------- T __nv_double2ull_rd
//-------- T __nv_double2ull_rn
//-------- T __nv_double2ull_ru
//-------- T __nv_double2ull_rz
CONVERT2ULL(double)

//-------- T __nv_float2ull_rd
//-------- T __nv_float2ull_rn
//-------- T __nv_float2ull_ru
//-------- T __nv_float2ull_rz
CONVERT2ULL(float)

#define CONVERT4LLM(A,B,m,n) ATTR B __nv_ll2##B##_##m(long x) \
    { return convert_##B##_##n(x); }

#define CONVERT4LL(B) \
    CONVERT4LLM(long, B, rd, rtn) \
    CONVERT4LLM(long, B, rn, rte) \
    CONVERT4LLM(long, B, ru, rtp) \
    CONVERT4LLM(long, B, rz, rtz)

//-------- T __nv_ll2double_rd
//-------- T __nv_ll2double_rn
//-------- T __nv_ll2double_ru
//-------- T __nv_ll2double_rz
CONVERT4LL(double)

//-------- T __nv_ll2float_rd
//-------- T __nv_ll2float_rn
//-------- T __nv_ll2float_ru
//-------- T __nv_ll2float_rz
CONVERT4LL(float)

#define CONVERT4ULLM(A,B,m,n) ATTR B __nv_ull2##B##_##m(ulong x) \
    { return convert_##B##_##n(x); }

#define CONVERT4ULL(B) \
    CONVERT4ULLM(ulong, B, rd, rtn) \
    CONVERT4ULLM(ulong, B, rn, rte) \
    CONVERT4ULLM(ulong, B, ru, rtp) \
    CONVERT4ULLM(ulong, B, rz, rtz)

//-------- T __nv_ull2double_rd
//-------- T __nv_ull2double_rn
//-------- T __nv_ull2double_ru
//-------- T __nv_ull2double_rz
CONVERT4ULL(double)

//-------- T __nv_ull2float_rd
//-------- T __nv_ull2float_rn
//-------- T __nv_ull2float_ru
//-------- T __nv_ull2float_rz
CONVERT4ULL(float)

