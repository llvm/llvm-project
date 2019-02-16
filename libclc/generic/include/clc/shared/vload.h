#define _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, VEC_TYPE, WIDTH, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DECL VEC_TYPE vload##SUFFIX##WIDTH(size_t offset, const ADDR_SPACE MEM_TYPE *x);

#define _CLC_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, ADDR_SPACE) \
  _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##2, 2, ADDR_SPACE) \
  _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##3, 3, ADDR_SPACE) \
  _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##4, 4, ADDR_SPACE) \
  _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##8, 8, ADDR_SPACE) \
  _CLC_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##16, 16, ADDR_SPACE)

#define _CLC_VECTOR_VLOAD_PRIM3(SUFFIX, MEM_TYPE, PRIM_TYPE) \
  _CLC_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __private) \
  _CLC_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __local) \
  _CLC_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __constant) \
  _CLC_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __global)

#define _CLC_VECTOR_VLOAD_PRIM1(PRIM_TYPE) \
  _CLC_VECTOR_VLOAD_PRIM3(, PRIM_TYPE, PRIM_TYPE)

// Declare vector load prototypes
_CLC_VECTOR_VLOAD_PRIM1(char)
_CLC_VECTOR_VLOAD_PRIM1(uchar)
_CLC_VECTOR_VLOAD_PRIM1(short)
_CLC_VECTOR_VLOAD_PRIM1(ushort)
_CLC_VECTOR_VLOAD_PRIM1(int)
_CLC_VECTOR_VLOAD_PRIM1(uint)
_CLC_VECTOR_VLOAD_PRIM1(long)
_CLC_VECTOR_VLOAD_PRIM1(ulong)
_CLC_VECTOR_VLOAD_PRIM1(float)
_CLC_VECTOR_VLOAD_PRIM3(_half, half, float)
// Use suffix to declare aligned vloada_halfN
_CLC_VECTOR_VLOAD_PRIM3(a_half, half, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
  _CLC_VECTOR_VLOAD_PRIM1(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
  _CLC_VECTOR_VLOAD_PRIM1(half)
#endif

// Scalar vload_half also needs to be declared
_CLC_VLOAD_DECL(_half, half, float, , __constant)
_CLC_VLOAD_DECL(_half, half, float, , __global)
_CLC_VLOAD_DECL(_half, half, float, , __local)
_CLC_VLOAD_DECL(_half, half, float, , __private)

// Scalar vloada_half is not part of the specs but CTS expects it
_CLC_VLOAD_DECL(a_half, half, float, , __constant)
_CLC_VLOAD_DECL(a_half, half, float, , __global)
_CLC_VLOAD_DECL(a_half, half, float, , __local)
_CLC_VLOAD_DECL(a_half, half, float, , __private)

#undef _CLC_VLOAD_DECL
#undef _CLC_VECTOR_VLOAD_DECL
#undef _CLC_VECTOR_VLOAD_PRIM3
#undef _CLC_VECTOR_VLOAD_PRIM1
