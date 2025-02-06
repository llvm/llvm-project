#include <clc/clcmacro.h>
#include <clc/integer/clc_add_sat.h>
#include <clc/integer/clc_mad24.h>
#include <clc/integer/clc_mul_hi.h>
#include <clc/integer/clc_upsample.h>
#include <clc/integer/definitions.h>
#include <clc/internal/clc.h>
#include <clc/relational/clc_select.h>
#include <clc/shared/clc_clamp.h>

#define __CLC_CONVERT_TY(X, TY) __builtin_convertvector(X, TY)

// Macro for defining mad_sat variants for char/uchar/short/ushort
// FIXME: Once using __clc_convert_ty, can easily unify scalar and vector defs
#define __CLC_DEFINE_SIMPLE_MAD_SAT(TYPE, UP_TYPE, LIT_PREFIX)                 \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_mad_sat(TYPE x, TYPE y, TYPE z) {          \
    return __clc_clamp(                                                        \
        (UP_TYPE)__clc_mad24((UP_TYPE)x, (UP_TYPE)y, (UP_TYPE)z),              \
        (UP_TYPE)LIT_PREFIX##_MIN, (UP_TYPE)LIT_PREFIX##_MAX);                 \
  }

#define __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE, UP_TYPE, LIT_PREFIX)             \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_mad_sat(TYPE x, TYPE y, TYPE z) {          \
    UP_TYPE upscaled_mad = __clc_mad24(__CLC_CONVERT_TY(x, UP_TYPE),           \
                                       __CLC_CONVERT_TY(y, UP_TYPE),           \
                                       __CLC_CONVERT_TY(z, UP_TYPE));          \
    UP_TYPE clamped_mad = __clc_clamp(upscaled_mad, (UP_TYPE)LIT_PREFIX##_MIN, \
                                      (UP_TYPE)LIT_PREFIX##_MAX);              \
    return __CLC_CONVERT_TY(clamped_mad, TYPE);                                \
  }

#define __CLC_DEFINE_SIMPLE_MAD_SAT_ALL_TYS(TYPE, UP_TYPE, LIT_PREFIX)         \
  __CLC_DEFINE_SIMPLE_MAD_SAT(TYPE, UP_TYPE, LIT_PREFIX)                       \
  __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE##2, UP_TYPE##2, LIT_PREFIX)             \
  __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE##3, UP_TYPE##3, LIT_PREFIX)             \
  __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE##4, UP_TYPE##4, LIT_PREFIX)             \
  __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE##8, UP_TYPE##8, LIT_PREFIX)             \
  __CLC_DEFINE_SIMPLE_MAD_SAT_VEC(TYPE##16, UP_TYPE##16, LIT_PREFIX)

__CLC_DEFINE_SIMPLE_MAD_SAT_ALL_TYS(char, int, CHAR)
__CLC_DEFINE_SIMPLE_MAD_SAT_ALL_TYS(uchar, uint, UCHAR)
__CLC_DEFINE_SIMPLE_MAD_SAT_ALL_TYS(short, int, SHRT)
__CLC_DEFINE_SIMPLE_MAD_SAT_ALL_TYS(ushort, uint, USHRT)

// Macro for defining mad_sat variants for uint/ulong
#define __CLC_DEFINE_UINTLONG_MAD_SAT(UTYPE, STYPE, ULIT_PREFIX)               \
  _CLC_OVERLOAD _CLC_DEF UTYPE __clc_mad_sat(UTYPE x, UTYPE y, UTYPE z) {      \
    STYPE has_mul_hi = __clc_mul_hi(x, y) != (UTYPE)0;                         \
    return __clc_select(__clc_add_sat(x * y, z), (UTYPE)ULIT_PREFIX##_MAX,     \
                        has_mul_hi);                                           \
  }

#define __CLC_DEFINE_UINTLONG_MAD_SAT_ALL_TYS(UTY, STY, ULIT_PREFIX)           \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY, STY, ULIT_PREFIX)                         \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY##2, STY##2, ULIT_PREFIX)                   \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY##3, STY##3, ULIT_PREFIX)                   \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY##4, STY##4, ULIT_PREFIX)                   \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY##8, STY##8, ULIT_PREFIX)                   \
  __CLC_DEFINE_UINTLONG_MAD_SAT(UTY##16, STY##16, ULIT_PREFIX)

__CLC_DEFINE_UINTLONG_MAD_SAT_ALL_TYS(uint, int, UINT)
__CLC_DEFINE_UINTLONG_MAD_SAT_ALL_TYS(ulong, long, ULONG)

// Macro for defining mad_sat variants for int
#define __CLC_DEFINE_SINT_MAD_SAT(INTTY, UINTTY, SLONGTY)                      \
  _CLC_OVERLOAD _CLC_DEF INTTY __clc_mad_sat(INTTY x, INTTY y, INTTY z) {      \
    INTTY mhi = __clc_mul_hi(x, y);                                            \
    UINTTY mlo = __clc_as_##UINTTY(x * y);                                     \
    SLONGTY m = __clc_upsample(mhi, mlo);                                      \
    m += __CLC_CONVERT_TY(z, SLONGTY);                                         \
    m = __clc_clamp(m, (SLONGTY)INT_MIN, (SLONGTY)INT_MAX);                    \
    return __CLC_CONVERT_TY(m, INTTY);                                         \
  }

// FIXME: Once using __clc_convert_ty, can easily unify scalar and vector defs
#define __CLC_DEFINE_SINT_MAD_SAT_ALL_TYS(INTTY, UINTTY, SLONGTY)              \
  _CLC_OVERLOAD _CLC_DEF INTTY __clc_mad_sat(INTTY x, INTTY y, INTTY z) {      \
    INTTY mhi = __clc_mul_hi(x, y);                                            \
    UINTTY mlo = __clc_as_##UINTTY(x * y);                                     \
    SLONGTY m = __clc_upsample(mhi, mlo);                                      \
    m += z;                                                                    \
    return __clc_clamp(m, (SLONGTY)INT_MIN, (SLONGTY)INT_MAX);                 \
  }                                                                            \
  __CLC_DEFINE_SINT_MAD_SAT(INTTY##2, UINTTY##2, SLONGTY##2)                   \
  __CLC_DEFINE_SINT_MAD_SAT(INTTY##3, UINTTY##3, SLONGTY##3)                   \
  __CLC_DEFINE_SINT_MAD_SAT(INTTY##4, UINTTY##4, SLONGTY##4)                   \
  __CLC_DEFINE_SINT_MAD_SAT(INTTY##8, UINTTY##8, SLONGTY##8)                   \
  __CLC_DEFINE_SINT_MAD_SAT(INTTY##16, UINTTY##16, SLONGTY##16)

__CLC_DEFINE_SINT_MAD_SAT_ALL_TYS(int, uint, long)

// Macro for defining mad_sat variants for long
#define __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY, ULONGTY)                           \
  _CLC_OVERLOAD _CLC_DEF SLONGTY __clc_mad_sat(SLONGTY x, SLONGTY y,           \
                                               SLONGTY z) {                    \
    SLONGTY hi = __clc_mul_hi(x, y);                                           \
    ULONGTY ulo = __clc_as_##ULONGTY(x * y);                                   \
    SLONGTY max1 = (x < 0) == (y < 0) && hi != 0;                              \
    SLONGTY max2 = hi == 0 && ulo >= LONG_MAX &&                               \
                   (z > 0 || (ulo + __clc_as_##ULONGTY(z)) > LONG_MAX);        \
    SLONGTY min1 = (((x < 0) != (y < 0)) && hi != -1);                         \
    SLONGTY min2 =                                                             \
        hi == -1 && ulo <= ((ULONGTY)LONG_MAX + 1UL) &&                        \
        (z < 0 || __clc_as_##ULONGTY(z) < ((ULONGTY)LONG_MAX - ulo));          \
    SLONGTY ret = __clc_as_##SLONGTY(ulo + __clc_as_##ULONGTY(z));             \
    ret = __clc_select(ret, (SLONGTY)LONG_MAX, (SLONGTY)(max1 || max2));       \
    ret = __clc_select(ret, (SLONGTY)LONG_MIN, (SLONGTY)(min1 || min2));       \
    return ret;                                                                \
  }

#define __CLC_DEFINE_SLONG_MAD_SAT_ALL_TYS(SLONGTY, ULONGTY)                   \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY, ULONGTY)                                 \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY##2, ULONGTY##2)                           \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY##3, ULONGTY##3)                           \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY##4, ULONGTY##4)                           \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY##8, ULONGTY##8)                           \
  __CLC_DEFINE_SLONG_MAD_SAT(SLONGTY##16, ULONGTY##16)

__CLC_DEFINE_SLONG_MAD_SAT_ALL_TYS(long, ulong)
