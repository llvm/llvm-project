/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Bitcasting

#define AS_SHORT(X) __builtin_astype(X, short)
#define AS_SHORT2(X) __builtin_astype(X, short2)
#define AS_USHORT(X) __builtin_astype(X, ushort)
#define AS_USHORT2(X) __builtin_astype(X, ushort2)
#define AS_INT(X) __builtin_astype(X, int)
#define AS_INT2(X) __builtin_astype(X, int2)
#define AS_UINT(X) __builtin_astype(X, uint)
#define AS_UINT2(X) __builtin_astype(X, uint2)
#define AS_LONG(X) __builtin_astype(X, long)
#define AS_ULONG(X) __builtin_astype(X, ulong)
#define AS_DOUBLE(X) __builtin_astype(X, double)
#define AS_FLOAT(X) __builtin_astype(X, float)
#define AS_HALF(X) __builtin_astype(X, half)
#define AS_HALF2(X) __builtin_astype(X, half2)

// Class mask bits
#define CLASS_SNAN __FPCLASS_SNAN
#define CLASS_QNAN __FPCLASS_QNAN
#define CLASS_NINF __FPCLASS_NEGINF
#define CLASS_NNOR __FPCLASS_NEGNORMAL
#define CLASS_NSUB __FPCLASS_NEGSUBNORMAL
#define CLASS_NZER __FPCLASS_NEGZERO
#define CLASS_PZER __FPCLASS_POSZERO
#define CLASS_PSUB __FPCLASS_POSSUBNORMAL
#define CLASS_PNOR __FPCLASS_POSNORMAL
#define CLASS_PINF __FPCLASS_POSINF

#include "irif.h"

#define BUILTIN_ABS_F32 __builtin_fabsf
#define BUILTIN_ABS_F64 __builtin_fabs
#define BUILTIN_ABS_F16 __builtin_fabsf16
#define BUILTIN_ABS_2F16 __builtin_elementwise_abs

#define BUILTIN_BITALIGN_B32 __builtin_amdgcn_alignbit

#define BUILTIN_CEIL_F32 __builtin_ceilf
#define BUILTIN_CEIL_F64 __builtin_ceil
#define BUILTIN_CEIL_F16 __builtin_ceilf16
#define BUILTIN_CEIL_2F16 __builtin_elementwise_ceil

#define BUILTIN_CLASS_F32 __builtin_isfpclass
#define BUILTIN_CLASS_F64 __builtin_isfpclass
#define BUILTIN_CLASS_F16 __builtin_isfpclass

#define BUILTIN_ISNAN_F32(x) __builtin_isnan(x)
#define BUILTIN_ISNAN_F64(x) __builtin_isnan(x)
#define BUILTIN_ISNAN_F16(x) __builtin_isnan(x)

#define BUILTIN_ISUNORDERED_F32(x, y) __builtin_isunordered(x, y)
#define BUILTIN_ISUNORDERED_F64(x, y) __builtin_isunordered(x, y)
#define BUILTIN_ISUNORDERED_F16(x, y) __builtin_isunordered(x, y)

#define BUILTIN_ISINF_F32(x) __builtin_isinf(x)
#define BUILTIN_ISINF_F64(x) __builtin_isinf(x)
#define BUILTIN_ISINF_F16(x) __builtin_isinf(x)

#define BUILTIN_ISFINITE_F32(x) __builtin_isfinite(x)
#define BUILTIN_ISFINITE_F64(x) __builtin_isfinite(x)
#define BUILTIN_ISFINITE_F16(x) __builtin_isfinite(x)

#define BUILTIN_ISSUBNORMAL_F32(x) __builtin_isfpclass(x, CLASS_NSUB|CLASS_PSUB)
#define BUILTIN_ISSUBNORMAL_F64(x) __builtin_isfpclass(x, CLASS_NSUB|CLASS_PSUB)
#define BUILTIN_ISSUBNORMAL_F16(x) __builtin_isfpclass(x, CLASS_NSUB|CLASS_PSUB)

#define BUILTIN_ISZERO_F32(x) __builtin_isfpclass(x, CLASS_NZER|CLASS_PZER)
#define BUILTIN_ISZERO_F64(x) __builtin_isfpclass(x, CLASS_NZER|CLASS_PZER)
#define BUILTIN_ISZERO_F16(x) __builtin_isfpclass(x, CLASS_NZER|CLASS_PZER)

#define BUILTIN_ISNORMAL_F32(x) __builtin_isnormal(x)
#define BUILTIN_ISNORMAL_F64(x) __builtin_isnormal(x)
#define BUILTIN_ISNORMAL_F16(x) __builtin_isnormal(x)

#define BUILTIN_COPYSIGN_F32 __builtin_copysignf
#define BUILTIN_COPYSIGN_F64 __builtin_copysign
#define BUILTIN_COPYSIGN_F16 __builtin_copysignf16
#define BUILTIN_COPYSIGN_2F16 __builtin_elementwise_copysign

#define BUILTIN_FLOOR_F32 __builtin_floorf
#define BUILTIN_FLOOR_F64 __builtin_floor
#define BUILTIN_FLOOR_F16 __builtin_floorf16
#define BUILTIN_FLOOR_2F16 __builtin_elementwise_floor

// These will codegen to v_fract_{f16|f32|f64} as appropriate.
#define BUILTIN_FRACTION_F32(X) ({                              \
    const float _x = X;                                         \
    const float _floor_x = BUILTIN_FLOOR_F32(_x);               \
    float _f = BUILTIN_MIN_F32(_x - _floor_x, 0x1.fffffep-1f);  \
    if (!FINITE_ONLY_OPT()) {                                   \
        _f = BUILTIN_ISNAN_F32(_x) ? _x : _f;                   \
        _f = BUILTIN_ISINF_F32(_x) ? 0.0f : _f;                  \
    }                                                           \
    _f;                                                         \
})

#define BUILTIN_FRACTION_F64(X) ({                                      \
    const double _x = X;                                                \
    const double _floor_x = BUILTIN_FLOOR_F64(_x);                       \
    double _f = BUILTIN_MIN_F64(_x - _floor_x, 0x1.fffffffffffffp-1);   \
    if (!FINITE_ONLY_OPT()) {                                           \
        _f = BUILTIN_ISNAN_F64(_x) ? _x : _f;                            \
        _f = BUILTIN_ISINF_F64(_x) ? 0.0 : _f;                           \
    }                                                                   \
    _f;                                                                 \
})

#define BUILTIN_FRACTION_F16(X) ({                                      \
    const half _x = X;                                                  \
    const half _floor_x = BUILTIN_FLOOR_F16(_x);                        \
    half _f = BUILTIN_MIN_F16(_x - _floor_x, 0x1.ffcp-1h);              \
    if (!FINITE_ONLY_OPT()) {                                           \
        _f = BUILTIN_ISNAN_F16(_x) ? _x : _f;                           \
        _f = BUILTIN_ISINF_F16(_x) ? 0.0h : _f;                         \
    }                                                                   \
    _f;                                                                 \
})

#define BUILTIN_MAD_U32(A,B,C) ((A)*(B)+(C))

#define BUILTIN_MAX_F32 __builtin_fmaxf
#define BUILTIN_MAX_F64 __builtin_fmax
#define BUILTIN_MAX_F16 __builtin_fmaxf16
#define BUILTIN_MAX_2F16 __builtin_elementwise_max

#define BUILTIN_MAX_S32(A,B) ((A) < (B) ? (B) : (A))
#define BUILTIN_MAX_U32(A,B) ((A) < (B) ? (B) : (A))

#define BUILTIN_MIN_F32 __builtin_fminf
#define BUILTIN_MIN_F64 __builtin_fmin
#define BUILTIN_MIN_F16 __builtin_fminf16
#define BUILTIN_MIN_2F16 __builtin_elementwise_min

#define BUILTIN_MIN_S32(A,B) ((A) < (B) ? (A) : (B))
#define BUILTIN_MIN_U32(A,B) ((A) < (B) ? (A) : (B))

#define BUILTIN_CANONICALIZE_F32(X) __builtin_canonicalizef(X)
#define BUILTIN_CANONICALIZE_F64(X) __builtin_canonicalize(X)
#define BUILTIN_CANONICALIZE_F16(X) __builtin_canonicalizef16(X)

#define BUILTIN_MULHI_U32(A,B) (((ulong)(A) * (ulong)(B)) >> 32)

#define BUILTIN_AMDGPU_COS_F32 __builtin_amdgcn_cosf

#define BUILTIN_AMDGPU_EXP2_F32 __builtin_amdgcn_exp2f
#define BUILTIN_EXP2_F32 __builtin_exp2f
#define BUILTIN_EXP2_F16 __builtin_exp2f16

#define BUILTIN_EXP_F32 __builtin_expf

#define BUILTIN_AMDGPU_LOG2_F32 __builtin_amdgcn_logf
#define BUILTIN_LOG2_F32 __builtin_log2f
#define BUILTIN_LOG2_F16 __builtin_log2f16

#define BUILTIN_LOG_F32 __builtin_logf
#define BUILTIN_LOG10_F32 __builtin_log10f

#define BUILTIN_AMDGPU_RCP_F32 __builtin_amdgcn_rcpf
#define BUILTIN_AMDGPU_RCP_F64 __builtin_amdgcn_rcp
#define BUILTIN_RCP_F16(X) (1.0h / (X))

#define BUILTIN_AMDGPU_RSQRT_F32 __builtin_amdgcn_rsqf
#define BUILTIN_AMDGPU_RSQRT_F64 __builtin_amdgcn_rsq
#define BUILTIN_RSQRT_F16(X) (1.0h / __builtin_sqrtf16(X))

#define BUILTIN_AMDGPU_SIN_F32 __builtin_amdgcn_sinf

#define BUILTIN_RINT_F32 __builtin_rintf
#define BUILTIN_RINT_F64 __builtin_rint
#define BUILTIN_RINT_F16 __builtin_rintf16
#define BUILTIN_RINT_2F16 __builtin_elementwise_rint

#define BUILTIN_SQRT_F32(X) __builtin_sqrtf(X)
#define BUILTIN_SQRT_F64(X) __builtin_sqrt(X)
#define BUILTIN_SQRT_F16(X) __builtin_sqrtf16(X)

#define BUILTIN_AMDGPU_SQRT_F32(X) __builtin_amdgcn_sqrtf(X)

#define BUILTIN_TRUNC_F32 __builtin_truncf
#define BUILTIN_TRUNC_F64 __builtin_trunc
#define BUILTIN_TRUNC_F16 __builtin_truncf16
#define BUILTIN_TRUNC_2F16 __builtin_elementwise_trunc

#define BUILTIN_ROUND_F32 __builtin_roundf
#define BUILTIN_ROUND_F64 __builtin_round
#define BUILTIN_ROUND_F16 __builtin_roundf16
#define BUILTIN_ROUND_2F16 __builtin_elementwise_round

#define BUILTIN_DIV_F32(X,Y) ({ \
    float _div_x = X; \
    float _div_y = Y; \
    float _div_ret = _div_x / _div_y; \
    _div_ret; \
})

#define BUILTIN_DIV_F64(X,Y) ({ \
    double _div_x = X; \
    double _div_y = Y; \
    double _div_ret = _div_x / _div_y; \
    _div_ret; \
})

#define BUILTIN_DIV_F16(X,Y) ({ \
    half _div_x = X; \
    half _div_y = Y; \
    half _div_ret = _div_x / _div_y; \
    _div_ret; \
})

#define BUILTIN_FMA_F32 __builtin_fmaf
#define BUILTIN_FMA_2F32 __builtin_elementwise_fma
#define BUILTIN_FMA_F64 __builtin_fma
#define BUILTIN_FMA_F16 __builtin_fmaf16
#define BUILTIN_FMA_2F16 __builtin_elementwise_fma

#define BUILTIN_FLDEXP_F32 __builtin_ldexpf
#define BUILTIN_FLDEXP_F64 __builtin_ldexp
#define BUILTIN_FLDEXP_F16 __builtin_ldexpf16

#define BUILTIN_FREXP_F32 __builtin_frexpf
#define BUILTIN_FREXP_F64 __builtin_frexp
#define BUILTIN_FREXP_F16 __builtin_frexpf16

#define BUILTIN_FREXP_EXP_F32(X)                                               \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexp(X, &_exp);                                             \
        _exp;                                                                  \
    })

#define BUILTIN_FREXP_EXP_F64(X)                                               \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexp(X, &_exp);                                             \
        _exp;                                                                  \
    })

#define BUILTIN_FREXP_EXP_F16(X)                                               \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexpf16(X, &_exp);                                          \
        _exp;                                                                  \
    })

#define BUILTIN_FREXP_MANT_F32(X)                                              \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexpf(X, &_exp);                                            \
    })

#define BUILTIN_FREXP_MANT_F64(X)                                              \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexp(X, &_exp);                                             \
    })

#define BUILTIN_FREXP_MANT_F16(X)                                              \
    ({                                                                         \
        int _exp;                                                              \
        __builtin_frexpf16(X, &_exp);                                          \
    })

#define BUILTIN_CMAX_F32 __builtin_fmaxf
#define BUILTIN_CMAX_F64 __builtin_fmax
#define BUILTIN_CMAX_F16 __builtin_fmaxf16
#define BUILTIN_CMAX_2F16 __builtin_elementwise_max

#define BUILTIN_CMIN_F32 __builtin_fminf
#define BUILTIN_CMIN_F64 __builtin_fmin
#define BUILTIN_CMIN_F16 __builtin_fminf16
#define BUILTIN_CMIN_2F16 __builtin_elementwise_min

#define BUILTIN_AMDGPU_TRIG_PREOP_F64 __builtin_amdgcn_trig_preop

#define BUILTIN_MAD_F32 __ocml_fmuladd_f32
#define BUILTIN_MAD_2F32 __ocml_fmuladd_2f32
#define BUILTIN_MAD_F64 __ocml_fmuladd_f64
#define BUILTIN_MAD_F16 __ocml_fmuladd_f16
#define BUILTIN_MAD_2F16 __ocml_fmuladd_2f16

// HW has ISA for max3, median3, and min3, median3 can be used to clamp
#define BUILTIN_CLAMP_S32(X,L,H) ({ \
    int _clamp_x = X; \
    int _clamp_l = L; \
    int _clamp_h = H; \
    int _clamp_r = _clamp_x > _clamp_l ? _clamp_x : _clamp_l; \
    _clamp_r = _clamp_r < _clamp_h ? _clamp_r : _clamp_h; \
    _clamp_r; \
})

#define BUILTIN_CLAMP_F32(X,L,H) __builtin_amdgcn_fmed3f(X,L,H)

#define ROUND_RTE 0
#define ROUND_RTP 1
#define ROUND_RTN 2
#define ROUND_RTZ 3

#define BUILTIN_GETROUND_F32() __builtin_amdgcn_s_getreg((1 << 0) | (0 << 6) | ((2-1) << 11))
#define BUILTIN_SETROUND_F32(X) __builtin_amdgcn_s_setreg((1 << 0) | (0 << 6) | ((2-1) << 11), X)
#define BUILTIN_GETROUND_F16F64() __builtin_amdgcn_s_getreg((1 << 0) | (2 << 6) | ((2-1) << 11))
#define BUILTIN_SETROUND_F16F64(X) __builtin_amdgcn_s_setreg((1 << 0) | (2 << 6) | ((2-1) << 11), X)
