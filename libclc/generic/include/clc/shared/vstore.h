#define _CLC_VSTORE_DECL(SUFFIX, PRIM_TYPE, VEC_TYPE, WIDTH, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DECL void vstore##SUFFIX##WIDTH(VEC_TYPE vec, size_t offset, ADDR_SPACE PRIM_TYPE *out);

#define _CLC_VECTOR_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, ADDR_SPACE) \
  _CLC_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##2, 2, ADDR_SPACE) \
  _CLC_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##3, 3, ADDR_SPACE) \
  _CLC_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##4, 4, ADDR_SPACE) \
  _CLC_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##8, 8, ADDR_SPACE) \
  _CLC_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##16, 16, ADDR_SPACE)

#define _CLC_VECTOR_VSTORE_PRIM3(SUFFIX, MEM_TYPE, PRIM_TYPE) \
  _CLC_VECTOR_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __private) \
  _CLC_VECTOR_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __local) \
  _CLC_VECTOR_VSTORE_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __global) \

#define _CLC_VECTOR_VSTORE_PRIM1(PRIM_TYPE) \
  _CLC_VECTOR_VSTORE_PRIM3(,PRIM_TYPE, PRIM_TYPE) \

_CLC_VECTOR_VSTORE_PRIM1(char)
_CLC_VECTOR_VSTORE_PRIM1(uchar)
_CLC_VECTOR_VSTORE_PRIM1(short)
_CLC_VECTOR_VSTORE_PRIM1(ushort)
_CLC_VECTOR_VSTORE_PRIM1(int)
_CLC_VECTOR_VSTORE_PRIM1(uint)
_CLC_VECTOR_VSTORE_PRIM1(long)
_CLC_VECTOR_VSTORE_PRIM1(ulong)
_CLC_VECTOR_VSTORE_PRIM1(float)
_CLC_VECTOR_VSTORE_PRIM3(_half, half, float)
// Use suffix to declare aligned vstorea_halfN
_CLC_VECTOR_VSTORE_PRIM3(a_half, half, float)

#ifdef cl_khr_fp64
  _CLC_VECTOR_VSTORE_PRIM1(double)
  _CLC_VECTOR_VSTORE_PRIM3(_half, half, double)
  // Use suffix to declare aligned vstorea_halfN
  _CLC_VECTOR_VSTORE_PRIM3(a_half, half, double)

  // Scalar vstore_half also needs to be declared
  _CLC_VSTORE_DECL(_half, half, double, , __private)
  _CLC_VSTORE_DECL(_half, half, double, , __local)
  _CLC_VSTORE_DECL(_half, half, double, , __global)

  // Scalar vstorea_half is not part of the specs but CTS expects it
  _CLC_VSTORE_DECL(a_half, half, double, , __private)
  _CLC_VSTORE_DECL(a_half, half, double, , __local)
  _CLC_VSTORE_DECL(a_half, half, double, , __global)
#endif

#ifdef cl_khr_fp16
  _CLC_VECTOR_VSTORE_PRIM1(half)
#endif

// Scalar vstore_half also needs to be declared
_CLC_VSTORE_DECL(_half, half, float, , __private)
_CLC_VSTORE_DECL(_half, half, float, , __local)
_CLC_VSTORE_DECL(_half, half, float, , __global)

// Scalar vstorea_half is not part of the specs but CTS expects it
_CLC_VSTORE_DECL(a_half, half, float, , __private)
_CLC_VSTORE_DECL(a_half, half, float, , __local)
_CLC_VSTORE_DECL(a_half, half, float, , __global)


#undef _CLC_VSTORE_DECL
#undef _CLC_VECTOR_VSTORE_DECL
#undef _CLC_VECTOR_VSTORE_PRIM3
#undef _CLC_VECTOR_VSTORE_PRIM1
