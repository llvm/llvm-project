#ifndef __CLC_CLCTYPES_H_
#define __CLC_CLCTYPES_H_

/* 6.1.1 Built-in Scalar Data Types */

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

#define __stdint_join3(a, b, c) a##b##c

#define __intn_t(n) __stdint_join3(__INT, n, _TYPE__)
#define __uintn_t(n) __stdint_join3(unsigned __INT, n, _TYPE__)

typedef __intn_t(__INTPTR_WIDTH__) intptr_t;
typedef __uintn_t(__INTPTR_WIDTH__) uintptr_t;

#undef __uintn_t
#undef __intn_t
#undef __stdint_join3

/* 6.1.2 Built-in Vector Data Types */

typedef __attribute__((ext_vector_type(2))) char char2;
typedef __attribute__((ext_vector_type(3))) char char3;
typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(8))) char char8;
typedef __attribute__((ext_vector_type(16))) char char16;

typedef __attribute__((ext_vector_type(2))) uchar uchar2;
typedef __attribute__((ext_vector_type(3))) uchar uchar3;
typedef __attribute__((ext_vector_type(4))) uchar uchar4;
typedef __attribute__((ext_vector_type(8))) uchar uchar8;
typedef __attribute__((ext_vector_type(16))) uchar uchar16;

typedef __attribute__((ext_vector_type(2))) short short2;
typedef __attribute__((ext_vector_type(3))) short short3;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(8))) short short8;
typedef __attribute__((ext_vector_type(16))) short short16;

typedef __attribute__((ext_vector_type(2))) ushort ushort2;
typedef __attribute__((ext_vector_type(3))) ushort ushort3;
typedef __attribute__((ext_vector_type(4))) ushort ushort4;
typedef __attribute__((ext_vector_type(8))) ushort ushort8;
typedef __attribute__((ext_vector_type(16))) ushort ushort16;

typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(3))) int int3;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(8))) int int8;
typedef __attribute__((ext_vector_type(16))) int int16;

typedef __attribute__((ext_vector_type(2))) uint uint2;
typedef __attribute__((ext_vector_type(3))) uint uint3;
typedef __attribute__((ext_vector_type(4))) uint uint4;
typedef __attribute__((ext_vector_type(8))) uint uint8;
typedef __attribute__((ext_vector_type(16))) uint uint16;

typedef __attribute__((ext_vector_type(2))) long long2;
typedef __attribute__((ext_vector_type(3))) long long3;
typedef __attribute__((ext_vector_type(4))) long long4;
typedef __attribute__((ext_vector_type(8))) long long8;
typedef __attribute__((ext_vector_type(16))) long long16;

typedef __attribute__((ext_vector_type(2))) ulong ulong2;
typedef __attribute__((ext_vector_type(3))) ulong ulong3;
typedef __attribute__((ext_vector_type(4))) ulong ulong4;
typedef __attribute__((ext_vector_type(8))) ulong ulong8;
typedef __attribute__((ext_vector_type(16))) ulong ulong16;

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(3))) float float3;
typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(8))) float float8;
typedef __attribute__((ext_vector_type(16))) float float16;

/* 9.3 Double Precision Floating-Point */

#ifdef cl_khr_fp64
typedef __attribute__((ext_vector_type(2))) double double2;
typedef __attribute__((ext_vector_type(3))) double double3;
typedef __attribute__((ext_vector_type(4))) double double4;
typedef __attribute__((ext_vector_type(8))) double double8;
typedef __attribute__((ext_vector_type(16))) double double16;
#endif

#ifdef cl_khr_fp16
typedef __attribute__((ext_vector_type(2))) half half2;
typedef __attribute__((ext_vector_type(3))) half half3;
typedef __attribute__((ext_vector_type(4))) half half4;
typedef __attribute__((ext_vector_type(8))) half half8;
typedef __attribute__((ext_vector_type(16))) half half16;
#endif

#endif // __CLC_CLCTYPES_H_
