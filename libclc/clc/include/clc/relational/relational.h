#ifndef __CLC_RELATIONAL_RELATIONAL_H__
#define __CLC_RELATIONAL_RELATIONAL_H__

/*
 * Contains relational macros that have to return 1 for scalar and -1 for vector
 * when the result is true.
 */

#define _CLC_DEFINE_SIMPLE_RELATIONAL_BINARY(RET_TYPE, RET_TYPE_VEC, FUNCTION, \
                                             ARG1_TYPE, ARG2_TYPE)             \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE_VEC##2 FUNCTION(ARG1_TYPE##2 x,              \
                                                  ARG2_TYPE##2 y) {            \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE_VEC##3 FUNCTION(ARG1_TYPE##3 x,              \
                                                  ARG2_TYPE##3 y) {            \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE_VEC##4 FUNCTION(ARG1_TYPE##4 x,              \
                                                  ARG2_TYPE##4 y) {            \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE_VEC##8 FUNCTION(ARG1_TYPE##8 x,              \
                                                  ARG2_TYPE##8 y) {            \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE_VEC##16 FUNCTION(ARG1_TYPE##16 x,            \
                                                   ARG2_TYPE##16 y) {          \
    return _CLC_RELATIONAL_OP(x, y);                                           \
  }

#define fcNan (__FPCLASS_SNAN | __FPCLASS_QNAN)
#define fcInf (__FPCLASS_POSINF | __FPCLASS_NEGINF)
#define fcNormal (__FPCLASS_POSNORMAL | __FPCLASS_NEGNORMAL)
#define fcPosFinite                                                            \
  (__FPCLASS_POSNORMAL | __FPCLASS_POSSUBNORMAL | __FPCLASS_POSZERO)
#define fcNegFinite                                                            \
  (__FPCLASS_NEGNORMAL | __FPCLASS_NEGSUBNORMAL | __FPCLASS_NEGZERO)
#define fcFinite (fcPosFinite | fcNegFinite)

#define _CLC_DEFINE_ISFPCLASS_VEC(RET_TYPE, FUNCTION, MASK, ARG_TYPE)          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (RET_TYPE)(__builtin_isfpclass(x, (MASK)) != (RET_TYPE)0);          \
  }

#define _CLC_DEFINE_ISFPCLASS(RET_TYPE, VEC_RET_TYPE, FUNCTION, MASK,          \
                              ARG_TYPE)                                        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return __builtin_isfpclass(x, (MASK));                                     \
  }                                                                            \
  _CLC_DEFINE_ISFPCLASS_VEC(VEC_RET_TYPE##2, FUNCTION, MASK, ARG_TYPE##2)      \
  _CLC_DEFINE_ISFPCLASS_VEC(VEC_RET_TYPE##3, FUNCTION, MASK, ARG_TYPE##3)      \
  _CLC_DEFINE_ISFPCLASS_VEC(VEC_RET_TYPE##4, FUNCTION, MASK, ARG_TYPE##4)      \
  _CLC_DEFINE_ISFPCLASS_VEC(VEC_RET_TYPE##8, FUNCTION, MASK, ARG_TYPE##8)      \
  _CLC_DEFINE_ISFPCLASS_VEC(VEC_RET_TYPE##16, FUNCTION, MASK, ARG_TYPE##16)

#endif // __CLC_RELATIONAL_RELATIONAL_H__
