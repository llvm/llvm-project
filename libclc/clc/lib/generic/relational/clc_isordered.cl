#include <clc/internal/clc.h>
#include <clc/relational/clc_isequal.h>
#include <clc/relational/relational.h>

#define _CLC_DEFINE_ISORDERED(RET_TYPE, FUNCTION, ARG1_TYPE, ARG2_TYPE)        \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return __clc_isequal(x, x) && __clc_isequal(y, y);                         \
  }

_CLC_DEFINE_ISORDERED(int, __clc_isordered, float, float)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(int, __clc_isordered, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isordered(double, double) returns an int, but the
// vector versions return long.

_CLC_DEFINE_ISORDERED(int, __clc_isordered, double, double)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, __clc_isordered, double, double)

#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isordered(half, half) returns an int, but the
// vector versions return short.

_CLC_DEFINE_ISORDERED(int, __clc_isordered, half, half)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, __clc_isordered, half, half)

#endif

#undef _CLC_DEFINE_ISORDERED
