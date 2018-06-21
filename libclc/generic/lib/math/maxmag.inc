#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
#define __CLC_CONVERT_NATN __CLC_XCONCAT(convert_long, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define __CLC_CONVERT_NATN __CLC_XCONCAT(convert_int, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define __CLC_CONVERT_NATN __CLC_XCONCAT(convert_short, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE maxmag(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  const __CLC_GENTYPE res = select(y, x, __CLC_CONVERT_NATN(isgreater(fabs(x), fabs(y))));
  return select(res, fmax(x, y), __CLC_CONVERT_NATN(isnan(x) | isnan(y) | isequal(fabs(x), fabs(y))));
}

#undef __CLC_CONVERT_NATN

#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
