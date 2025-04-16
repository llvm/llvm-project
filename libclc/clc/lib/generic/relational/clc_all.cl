#include <clc/internal/clc.h>

#define _CLC_ALL(v) (((v) >> ((sizeof(v) * 8) - 1)) & 0x1)

#define _CLC_ALL_VEC(TYPE)                                                     \
  _CLC_OVERLOAD _CLC_DEF int __clc_all(TYPE v) {                               \
    return _CLC_ALL(__builtin_reduce_and(v));                                  \
  }

#define _CLC_DEFINE_ALL(TYPE)                                                  \
  _CLC_OVERLOAD _CLC_DEF int __clc_all(TYPE v) { return _CLC_ALL(v); }         \
  _CLC_ALL_VEC(TYPE##2)                                                        \
  _CLC_ALL_VEC(TYPE##3)                                                        \
  _CLC_ALL_VEC(TYPE##4)                                                        \
  _CLC_ALL_VEC(TYPE##8)                                                        \
  _CLC_ALL_VEC(TYPE##16)

_CLC_DEFINE_ALL(char)
_CLC_DEFINE_ALL(short)
_CLC_DEFINE_ALL(int)
_CLC_DEFINE_ALL(long)
