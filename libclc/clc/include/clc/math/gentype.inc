#include <clc/clcfunc.h>
#include <clc/clctypes.h>
#include <clc/utils.h>

// Define some useful macros for type conversions.
#define __CLC_AS_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_GENTYPE)
#define __CLC_CONVERT_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)

// Define some macros for types matching the same vector size as __CLC_GENTYPE.
#define __CLC_HALFN __CLC_XCONCAT(half, __CLC_VECSIZE)
#define __CLC_FLOATN __CLC_XCONCAT(float, __CLC_VECSIZE)
#define __CLC_DOUBLEN __CLC_XCONCAT(double, __CLC_VECSIZE)

#define __CLC_CHARN __CLC_XCONCAT(char, __CLC_VECSIZE)
#define __CLC_SHORTN __CLC_XCONCAT(short, __CLC_VECSIZE)
#define __CLC_INTN __CLC_XCONCAT(int, __CLC_VECSIZE)
#define __CLC_LONGN __CLC_XCONCAT(long, __CLC_VECSIZE)

#define __CLC_UCHARN __CLC_XCONCAT(uchar, __CLC_VECSIZE)
#define __CLC_USHORTN __CLC_XCONCAT(ushort, __CLC_VECSIZE)
#define __CLC_UINTN __CLC_XCONCAT(uint, __CLC_VECSIZE)
#define __CLC_ULONGN __CLC_XCONCAT(ulong, __CLC_VECSIZE)

#define __CLC_AS_CHARN __CLC_XCONCAT(__clc_as_, __CLC_CHARN)
#define __CLC_AS_SHORTN __CLC_XCONCAT(__clc_as_, __CLC_SHORTN)
#define __CLC_AS_INTN __CLC_XCONCAT(__clc_as_, __CLC_INTN)
#define __CLC_AS_LONGN __CLC_XCONCAT(__clc_as_, __CLC_LONGN)

#define __CLC_AS_UCHARN __CLC_XCONCAT(__clc_as_, __CLC_UCHARN)
#define __CLC_AS_USHORTN __CLC_XCONCAT(__clc_as_, __CLC_USHORTN)
#define __CLC_AS_UINTN __CLC_XCONCAT(__clc_as_, __CLC_UINTN)
#define __CLC_AS_ULONGN __CLC_XCONCAT(__clc_as_, __CLC_ULONGN)

#define __CLC_CONVERT_HALFN __CLC_XCONCAT(__clc_convert_half, __CLC_VECSIZE)
#define __CLC_CONVERT_FLOATN __CLC_XCONCAT(__clc_convert_float, __CLC_VECSIZE)
#define __CLC_CONVERT_DOUBLEN __CLC_XCONCAT(__clc_convert_double, __CLC_VECSIZE)

#define __CLC_CONVERT_CHARN __CLC_XCONCAT(__clc_convert_, __CLC_CHARN)
#define __CLC_CONVERT_SHORTN __CLC_XCONCAT(__clc_convert_, __CLC_SHORTN)
#define __CLC_CONVERT_INTN __CLC_XCONCAT(__clc_convert_, __CLC_INTN)
#define __CLC_CONVERT_LONGN __CLC_XCONCAT(__clc_convert_, __CLC_LONGN)

#define __CLC_CONVERT_UCHARN __CLC_XCONCAT(__clc_convert_, __CLC_UCHARN)
#define __CLC_CONVERT_USHORTN __CLC_XCONCAT(__clc_convert_, __CLC_USHORTN)
#define __CLC_CONVERT_UINTN __CLC_XCONCAT(__clc_convert_, __CLC_UINTN)
#define __CLC_CONVERT_ULONGN __CLC_XCONCAT(__clc_convert_, __CLC_ULONGN)

// See definitions of __CLC_S_GENTYPE/__CLC_U_GENTYPE below, which depend on the
// specific size of floating-point type. These are the signed and unsigned
// integers of the same bitwidth and element count as the GENTYPE. They match
// the naming conventions in the integer version gentype.inc, for
// convenience.
#define __CLC_AS_S_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_S_GENTYPE)
#define __CLC_AS_U_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_U_GENTYPE)

#define __CLC_CONVERT_S_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_S_GENTYPE)
#define __CLC_CONVERT_U_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_U_GENTYPE)

#define __CLC_SCALAR_GENTYPE float
#define __CLC_FPSIZE 32
#define __CLC_FP_LIT(x) x##F

#define __CLC_S_GENTYPE __CLC_XCONCAT(int, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(uint, __CLC_VECSIZE)

#define __CLC_GENTYPE float
#define __CLC_BIT_INTN int
#define __CLC_SCALAR
#define __CLC_VECSIZE
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN
#undef __CLC_SCALAR

#define __CLC_GENTYPE float2
#define __CLC_BIT_INTN int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE float3
#define __CLC_BIT_INTN int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE float4
#define __CLC_BIT_INTN int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE float8
#define __CLC_BIT_INTN int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE float16
#define __CLC_BIT_INTN int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_FP_LIT
#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_SCALAR_GENTYPE double
#define __CLC_FPSIZE 64
#define __CLC_FP_LIT(x) (x)

#define __CLC_S_GENTYPE __CLC_XCONCAT(long, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(ulong, __CLC_VECSIZE)

#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_GENTYPE double
#define __CLC_BIT_INTN long
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN
#undef __CLC_VECSIZE
#undef __CLC_SCALAR

#define __CLC_GENTYPE double2
#define __CLC_BIT_INTN long2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE double3
#define __CLC_BIT_INTN long3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE double4
#define __CLC_BIT_INTN long4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE double8
#define __CLC_BIT_INTN long8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE double16
#define __CLC_BIT_INTN long16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_FP_LIT
#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE
#endif
#endif

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_SCALAR_GENTYPE half
#define __CLC_FPSIZE 16
#define __CLC_FP_LIT(x) x##H

#define __CLC_S_GENTYPE __CLC_XCONCAT(short, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(ushort, __CLC_VECSIZE)

#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_GENTYPE half
#define __CLC_BIT_INTN short
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN
#undef __CLC_VECSIZE
#undef __CLC_SCALAR

#define __CLC_GENTYPE half2
#define __CLC_BIT_INTN short2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE half3
#define __CLC_BIT_INTN short3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE half4
#define __CLC_BIT_INTN short4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE half8
#define __CLC_BIT_INTN short8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#define __CLC_GENTYPE half16
#define __CLC_BIT_INTN short16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_BIT_INTN

#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_FP_LIT
#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE
#endif
#endif

#undef __CLC_BODY

#undef __CLC_AS_U_GENTYPE
#undef __CLC_AS_S_GENTYPE

#undef __CLC_CONVERT_U_GENTYPE
#undef __CLC_CONVERT_S_GENTYPE

#undef __CLC_AS_CHARN
#undef __CLC_AS_SHORTN
#undef __CLC_AS_INTN
#undef __CLC_AS_LONGN

#undef __CLC_AS_UCHARN
#undef __CLC_AS_USHORTN
#undef __CLC_AS_UINTN
#undef __CLC_AS_ULONGN

#undef __CLC_CONVERT_HALFN
#undef __CLC_CONVERT_FLOATN
#undef __CLC_CONVERT_DOUBLEN

#undef __CLC_CONVERT_CHARN
#undef __CLC_CONVERT_SHORTN
#undef __CLC_CONVERT_INTN
#undef __CLC_CONVERT_LONGN

#undef __CLC_CONVERT_UCHARN
#undef __CLC_CONVERT_USHORTN
#undef __CLC_CONVERT_UINTN
#undef __CLC_CONVERT_ULONGN

#undef __CLC_ULONGN
#undef __CLC_UINTN
#undef __CLC_USHORTN
#undef __CLC_UCHARN

#undef __CLC_LONGN
#undef __CLC_INTN
#undef __CLC_SHORTN
#undef __CLC_CHARN

#undef __CLC_DOUBLEN
#undef __CLC_FLOATN
#undef __CLC_HALFN

#undef __CLC_AS_GENTYPE
#undef __CLC_CONVERT_GENTYPE
