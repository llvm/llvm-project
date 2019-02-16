#if __CLC_FPSIZE == 32

#ifdef __CLC_VECSIZE

#if __CLC_VECSIZE == 3
#  define __CLC_OFFSET 4
#else
#  define __CLC_OFFSET __CLC_VECSIZE
#endif

  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __global);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __constant);

#undef __CLC_OFFSET
#else
  FUNC(, 1, 1, __CLC_GENTYPE, __private);
  FUNC(, 1, 1, __CLC_GENTYPE, __local);
  FUNC(, 1, 1, __CLC_GENTYPE, __global);
  FUNC(, 1, 1, __CLC_GENTYPE, __constant);
#endif
#endif
