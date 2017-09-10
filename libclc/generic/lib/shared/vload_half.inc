#if __CLC_FPSIZE == 32
#ifdef __CLC_VECSIZE
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __global);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __constant);
#else
  FUNC(, 1, __CLC_GENTYPE, __private);
  FUNC(, 1, __CLC_GENTYPE, __local);
  FUNC(, 1, __CLC_GENTYPE, __global);
  FUNC(, 1, __CLC_GENTYPE, __constant);
#endif
#endif
