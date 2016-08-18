
#ifdef __CLC_VECSIZE
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __global);
#else
  FUNC(, 1, __CLC_GENTYPE, __private);
  FUNC(, 1, __CLC_GENTYPE, __local);
  FUNC(, 1, __CLC_GENTYPE, __global);
#endif
