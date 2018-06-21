#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
#define __CLC_NATN __CLC_XCONCAT(ulong, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define __CLC_NATN __CLC_XCONCAT(uint, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define __CLC_NATN __CLC_XCONCAT(ushort, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE nan(__CLC_NATN code);

#undef __CLC_NATN
#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
