#ifndef __CLC_RELATIONAL_RELATIONAL_H__
#define __CLC_RELATIONAL_RELATIONAL_H__

/*
 * Contains relational macros that have to return 1 for scalar and -1 for vector
 * when the result is true.
 */

#define _CLC_DEFINE_RELATIONAL_UNARY_SCALAR(RET_TYPE, FUNCTION, BUILTIN_NAME,  \
                                            ARG_TYPE)                          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return BUILTIN_NAME(x);                                                    \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC2(RET_TYPE, FUNCTION, ARG_TYPE)        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.lo), FUNCTION(x.hi)} !=            \
                      (RET_TYPE)0);                                            \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC3(RET_TYPE, FUNCTION, ARG_TYPE)        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0), FUNCTION(x.s1),               \
                                 FUNCTION(x.s2)} != (RET_TYPE)0);              \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC4(RET_TYPE, FUNCTION, ARG_TYPE)        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0), FUNCTION(x.s1),               \
                                 FUNCTION(x.s2),                               \
                                 FUNCTION(x.s3)} != (RET_TYPE)0);              \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC8(RET_TYPE, FUNCTION, ARG_TYPE)        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (                                                                   \
        RET_TYPE)((RET_TYPE){FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2),   \
                             FUNCTION(x.s3), FUNCTION(x.s4), FUNCTION(x.s5),   \
                             FUNCTION(x.s6), FUNCTION(x.s7)} != (RET_TYPE)0);  \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC16(RET_TYPE, FUNCTION, ARG_TYPE)       \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG_TYPE x) {                       \
    return (                                                                   \
        RET_TYPE)((RET_TYPE){FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2),   \
                             FUNCTION(x.s3), FUNCTION(x.s4), FUNCTION(x.s5),   \
                             FUNCTION(x.s6), FUNCTION(x.s7), FUNCTION(x.s8),   \
                             FUNCTION(x.s9), FUNCTION(x.sa), FUNCTION(x.sb),   \
                             FUNCTION(x.sc), FUNCTION(x.sd), FUNCTION(x.se),   \
                             FUNCTION(x.sf)} != (RET_TYPE)0);                  \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(RET_TYPE, FUNCTION, ARG_TYPE)     \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC2(RET_TYPE##2, FUNCTION, ARG_TYPE##2)        \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC3(RET_TYPE##3, FUNCTION, ARG_TYPE##3)        \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC4(RET_TYPE##4, FUNCTION, ARG_TYPE##4)        \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC8(RET_TYPE##8, FUNCTION, ARG_TYPE##8)        \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC16(RET_TYPE##16, FUNCTION, ARG_TYPE##16)

#define _CLC_DEFINE_RELATIONAL_UNARY(RET_TYPE, FUNCTION, BUILTIN_FUNCTION,     \
                                     ARG_TYPE)                                 \
  _CLC_DEFINE_RELATIONAL_UNARY_SCALAR(RET_TYPE, FUNCTION, BUILTIN_FUNCTION,    \
                                      ARG_TYPE)                                \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(RET_TYPE, FUNCTION, ARG_TYPE)

#define _CLC_DEFINE_RELATIONAL_BINARY_SCALAR(RET_TYPE, FUNCTION, BUILTIN_NAME, \
                                             ARG0_TYPE, ARG1_TYPE)             \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return BUILTIN_NAME(x, y);                                                 \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC(RET_TYPE, FUNCTION, ARG0_TYPE,       \
                                          ARG1_TYPE)                           \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.lo, y.lo),                         \
                                 FUNCTION(x.hi, y.hi)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC2(RET_TYPE, FUNCTION, ARG0_TYPE,      \
                                           ARG1_TYPE)                          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.lo, y.lo),                         \
                                 FUNCTION(x.hi, y.hi)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC3(RET_TYPE, FUNCTION, ARG0_TYPE,      \
                                           ARG1_TYPE)                          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),   \
                                 FUNCTION(x.s2, y.s2)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC4(RET_TYPE, FUNCTION, ARG0_TYPE,      \
                                           ARG1_TYPE)                          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),   \
                                 FUNCTION(x.s2, y.s2),                         \
                                 FUNCTION(x.s3, y.s3)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC8(RET_TYPE, FUNCTION, ARG0_TYPE,      \
                                           ARG1_TYPE)                          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),   \
                                 FUNCTION(x.s2, y.s2), FUNCTION(x.s3, y.s3),   \
                                 FUNCTION(x.s4, y.s4), FUNCTION(x.s5, y.s5),   \
                                 FUNCTION(x.s6, y.s6),                         \
                                 FUNCTION(x.s7, y.s7)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC16(RET_TYPE, FUNCTION, ARG0_TYPE,     \
                                            ARG1_TYPE)                         \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG0_TYPE x, ARG1_TYPE y) {         \
    return (RET_TYPE)((RET_TYPE){FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),   \
                                 FUNCTION(x.s2, y.s2), FUNCTION(x.s3, y.s3),   \
                                 FUNCTION(x.s4, y.s4), FUNCTION(x.s5, y.s5),   \
                                 FUNCTION(x.s6, y.s6), FUNCTION(x.s7, y.s7),   \
                                 FUNCTION(x.s8, y.s8), FUNCTION(x.s9, y.s9),   \
                                 FUNCTION(x.sa, y.sa), FUNCTION(x.sb, y.sb),   \
                                 FUNCTION(x.sc, y.sc), FUNCTION(x.sd, y.sd),   \
                                 FUNCTION(x.se, y.se),                         \
                                 FUNCTION(x.sf, y.sf)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(RET_TYPE, FUNCTION, ARG0_TYPE,   \
                                              ARG1_TYPE)                       \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC2(RET_TYPE##2, FUNCTION, ARG0_TYPE##2,      \
                                     ARG1_TYPE##2)                             \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC3(RET_TYPE##3, FUNCTION, ARG0_TYPE##3,      \
                                     ARG1_TYPE##3)                             \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC4(RET_TYPE##4, FUNCTION, ARG0_TYPE##4,      \
                                     ARG1_TYPE##4)                             \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC8(RET_TYPE##8, FUNCTION, ARG0_TYPE##8,      \
                                     ARG1_TYPE##8)                             \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC16(RET_TYPE##16, FUNCTION, ARG0_TYPE##16,   \
                                      ARG1_TYPE##16)

#define _CLC_DEFINE_RELATIONAL_BINARY(RET_TYPE, FUNCTION, BUILTIN_FUNCTION,    \
                                      ARG0_TYPE, ARG1_TYPE)                    \
  _CLC_DEFINE_RELATIONAL_BINARY_SCALAR(RET_TYPE, FUNCTION, BUILTIN_FUNCTION,   \
                                       ARG0_TYPE, ARG1_TYPE)                   \
  _CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(RET_TYPE, FUNCTION, ARG0_TYPE,         \
                                        ARG1_TYPE)

#endif // __CLC_RELATIONAL_RELATIONAL_H__
