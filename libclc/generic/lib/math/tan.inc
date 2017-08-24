/*
 * Note: tan(x) = sin(x)/cos(x) also, but the final assembly ends up being
 *       twice as long for R600 (maybe for others as well).
 */

#if __CLC_FPSIZE == 32
#define __CLC_CONST(x) x ## f
#else
#define __CLC_CONST(x) x
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE tan(__CLC_GENTYPE x) {
  __CLC_GENTYPE sinx = sin(x);
  return sinx / sqrt( (__CLC_GENTYPE) __CLC_CONST(1.0) - (sinx*sinx) );
}

#undef __CLC_CONST
