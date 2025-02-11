#include <clc/internal/clc.h>

#define _CLC_ANY(v) (((v) >> ((sizeof(v) * 8) - 1)) & 0x1)

#define _CLC_ANY_VEC(TYPE)                                                     \
  _CLC_OVERLOAD _CLC_DEF int __clc_any(TYPE v) {                               \
    return _CLC_ANY(__builtin_reduce_or(v));                                   \
  }

#define _CLC_DEFINE_ANY(TYPE)                                                  \
  _CLC_OVERLOAD _CLC_DEF int __clc_any(TYPE v) { return _CLC_ANY(v); }         \
  _CLC_ANY_VEC(TYPE##2)                                                        \
  _CLC_ANY_VEC(TYPE##3)                                                        \
  _CLC_ANY_VEC(TYPE##4)                                                        \
  _CLC_ANY_VEC(TYPE##8)                                                        \
  _CLC_ANY_VEC(TYPE##16)

_CLC_DEFINE_ANY(char)
_CLC_DEFINE_ANY(short)
_CLC_DEFINE_ANY(int)
_CLC_DEFINE_ANY(long)
