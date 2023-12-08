#include <clc/clc.h>

#define _CLC_ALL(v) (((v) >> ((sizeof(v) * 8) - 1)) & 0x1)
#define _CLC_ALL2(v) (_CLC_ALL((v).s0) & _CLC_ALL((v).s1))
#define _CLC_ALL3(v) (_CLC_ALL2((v)) & _CLC_ALL((v).s2))
#define _CLC_ALL4(v) (_CLC_ALL3((v)) & _CLC_ALL((v).s3))
#define _CLC_ALL8(v) (_CLC_ALL4((v)) & _CLC_ALL((v).s4) & _CLC_ALL((v).s5) \
                                     & _CLC_ALL((v).s6) & _CLC_ALL((v).s7))
#define _CLC_ALL16(v) (_CLC_ALL8((v)) & _CLC_ALL((v).s8) & _CLC_ALL((v).s9) \
                                      & _CLC_ALL((v).sA) & _CLC_ALL((v).sB) \
                                      & _CLC_ALL((v).sC) & _CLC_ALL((v).sD) \
                                      & _CLC_ALL((v).sE) & _CLC_ALL((v).sf))


#define ALL_ID(TYPE) \
  _CLC_OVERLOAD _CLC_DEF int all(TYPE v)

#define ALL_VECTORIZE(TYPE) \
  ALL_ID(TYPE) { return _CLC_ALL(v); } \
  ALL_ID(TYPE##2) { return _CLC_ALL2(v); } \
  ALL_ID(TYPE##3) { return _CLC_ALL3(v); } \
  ALL_ID(TYPE##4) { return _CLC_ALL4(v); } \
  ALL_ID(TYPE##8) { return _CLC_ALL8(v); } \
  ALL_ID(TYPE##16) { return _CLC_ALL16(v); }

ALL_VECTORIZE(char)
ALL_VECTORIZE(short)
ALL_VECTORIZE(int)
ALL_VECTORIZE(long)
