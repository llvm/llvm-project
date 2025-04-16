#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
#define __CLC_S_GENTYPE __CLC_XCONCAT(long, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(ulong, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define __CLC_S_GENTYPE __CLC_XCONCAT(int, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(uint, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define __CLC_S_GENTYPE __CLC_XCONCAT(short, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(ushort, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_SELECT_FN(__CLC_GENTYPE x,
                                                      __CLC_GENTYPE y,
                                                      __CLC_S_GENTYPE z);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_SELECT_FN(__CLC_GENTYPE x,
                                                      __CLC_GENTYPE y,
                                                      __CLC_U_GENTYPE z);

#ifdef __CLC_FPSIZE
#undef __CLC_S_GENTYPE
#undef __CLC_U_GENTYPE
#endif
#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
