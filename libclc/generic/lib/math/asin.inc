// TODO: Enable half precision when atan2 is implemented
#if __CLC_FPSIZE > 16

#if __CLC_FPSIZE == 64
#define __CLC_CONST(x) x
#elif __CLC_FPSIZE == 32
#define __CLC_CONST(x) x ## f
#elif __CLC_FPSIZE == 16
#define __CLC_CONST(x) x ## h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE asin(__CLC_GENTYPE x) {
  return atan2(x, sqrt( (__CLC_GENTYPE)__CLC_CONST(1.0) - (x*x) ));
}

#undef __CLC_CONST

#endif
