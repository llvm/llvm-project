
#if __CLC_FPSIZE == 32
#define __CLC_CONST(x) x ## f
#else
#define __CLC_CONST(x) x
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE asin(__CLC_GENTYPE x) {
  return atan2(x, sqrt( (__CLC_GENTYPE)__CLC_CONST(1.0) -(x*x) ));
}

#undef __CLC_CONST
