
#if !defined(__CLC_SCALAR)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fmax(__CLC_GENTYPE x, float y) {
  return fmax(x, (__CLC_GENTYPE)((__CLC_SCALAR_GENTYPE)y));
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fmax(__CLC_GENTYPE x, double y) {
  return fmax(x, (__CLC_GENTYPE)((__CLC_SCALAR_GENTYPE)y));
}

#endif // ifdef cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fmax(__CLC_GENTYPE x, half y) {
  return fmax(x, (__CLC_GENTYPE)((__CLC_SCALAR_GENTYPE)y));
}

#endif // ifdef cl_khr_fp16

#endif // !defined(__CLC_SCALAR)
