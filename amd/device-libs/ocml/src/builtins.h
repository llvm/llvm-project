
// Class mask bits
#define CLASS_SNAN 0x001
#define CLASS_QNAN 0x002
#define CLASS_NINF 0x004
#define CLASS_NNOR 0x008
#define CLASS_NSUB 0x010
#define CLASS_NZER 0x020
#define CLASS_PZER 0x040
#define CLASS_PSUB 0x080
#define CLASS_PNOR 0x100
#define CLASS_PINF 0x200

#include "llvm.h"

#define BUILTIN_ABS_F32 __llvm_fabs_f32
#define BUILTIN_ABS_F64 __llvm_fabs_f64
#define BUILTIN_ABS_F16 __llvm_fabs_f16

#define BUILTIN_BITALIGN_B32(A,B,C) ((((ulong)(A) << 32) | (ulong)(B)) >> ((C) & 0x1f))

#define BUILTIN_CEIL_F32 __llvm_ceil_f32
#define BUILTIN_CEIL_F64 __llvm_ceil_f64
#define BUILTIN_CEIL_F16 __llvm_ceil_f16

#define BUILTIN_CLASS_F32 __llvm_amdgcn_class_f32
#define BUILTIN_CLASS_F64 __llvm_amdgcn_class_f64
#define BUILTIN_CLASS_F16 __llvm_amdgcn_class_f16

#define BUILTIN_COPYSIGN_F32 __llvm_copysign_f32
#define BUILTIN_COPYSIGN_F64 __llvm_copysign_f64
#define BUILTIN_COPYSIGN_F16 __llvm_copysign_f16

#define BUILTIN_FIRSTBIT_U32(X) ((X) == 0 ? -1 : __builtin_clz(X))

#define BUILTIN_FLOOR_F32 __llvm_floor_f32
#define BUILTIN_FLOOR_F64 __llvm_floor_f64
#define BUILTIN_FLOOR_F16 __llvm_floor_f16

#define BUILTIN_FRACTION_F32 __llvm_amdgcn_fract_f32
#define BUILTIN_FRACTION_F64 __llvm_amdgcn_fract_f64
#define BUILTIN_FRACTION_F16 __llvm_amdgcn_fract_f16

#define BUILTIN_MAD_U32(A,B,C) ((A)*(B)+(C))

#define BUILTIN_MAX_F32 __llvm_maxnum_f32
#define BUILTIN_MAX_F64 __llvm_maxnum_f64
#define BUILTIN_MAX_F16 __llvm_maxnum_f16

#define BUILTIN_MAX_S32(A,B) ((A) < (B) ? (B) : (A))
#define BUILTIN_MAX_U32(A,B) ((A) < (B) ? (B) : (A))

#define BUILTIN_MIN_F32 __llvm_minnum_f32
#define BUILTIN_MIN_F64 __llvm_minnum_f64
#define BUILTIN_MIN_F16 __llvm_minnum_f16

#define BUILTIN_MIN_S32(A,B) ((A) < (B) ? (A) : (B))
#define BUILTIN_MIN_U32(A,B) ((A) < (B) ? (A) : (B))

#define BUILTIN_CANONICALIZE_F32(X) __llvm_canonicalize_f32(X)
#define BUILTIN_CANONICALIZE_F64(X) __llvm_canonicalize_f64(X)
#define BUILTIN_CANONICALIZE_F16(X) __llvm_canonicalize_f16(X)

#define BUILTIN_MULHI_U32(A,B) (((ulong)(A) * (ulong)(B)) >> 32)

#define BUILTIN_COS_F32 __llvm_amdgcn_cos_f32

#define BUILTIN_EXP2_F32 __llvm_amdgcn_exp_f32
#define BUILTIN_EXP2_F16 __llvm_amdgcn_exp_f16

#define BUILTIN_LOG2_F32 __llvm_amdgcn_log_f32
#define BUILTIN_LOG2_F16 __llvm_amdgcn_log_f16

#define BUILTIN_RCP_F32 __llvm_amdgcn_rcp_f32
#define BUILTIN_RCP_F64 __llvm_amdgcn_rcp_f64
#define BUILTIN_RCP_F16 __llvm_amdgcn_rcp_f16

#define BUILTIN_RSQRT_F32 __llvm_amdgcn_rsq_f32
#define BUILTIN_RSQRT_F64 __llvm_amdgcn_rsq_f64
#define BUILTIN_RSQRT_F16 __llvm_amdgcn_rsq_f16

#define BUILTIN_SIN_F32 __llvm_amdgcn_sin_f32

#define BUILTIN_RINT_F32 __llvm_rint_f32
#define BUILTIN_RINT_F64 __llvm_rint_f64
#define BUILTIN_RINT_F16 __llvm_rint_f16

#define BUILTIN_SQRT_F32(X) ({ \
    float _bsqrt_x = X; \
    float _bsqrt_ret = _bsqrt_x < 0.0 ? __builtin_nanf("") : __llvm_sqrt_f32(_bsqrt_x); \
    _bsqrt_ret; \
})

#define BUILTIN_SQRT_F64(X) ({ \
    double _bsqrt_x = X; \
    double _bsqrt_ret = _bsqrt_x < 0.0 ? __builtin_nan("") : __llvm_sqrt_f64(_bsqrt_x); \
    _bsqrt_ret; \
})

#define BUILTIN_SQRT_F16(X) ({ \
    float _bsqrt_x = X; \
    float _bsqrt_ret = _bsqrt_x < 0.0 ? (half)__builtin_nanf("") : __llvm_sqrt_f16(_bsqrt_x); \
    _bsqrt_ret; \
})

#define BUILTIN_TRUNC_F32 __llvm_trunc_f32
#define BUILTIN_TRUNC_F64 __llvm_trunc_f64
#define BUILTIN_TRUNC_F16 __llvm_trunc_f16

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

#define BUILTIN_FMA_F32 __llvm_fma_f32
#define BUILTIN_FMA_F64 __llvm_fma_f64
#define BUILTIN_FMA_F16 __llvm_fma_f16

#define BUILTIN_FLDEXP_F32 __llvm_amdgcn_ldexp_f32
#define BUILTIN_FLDEXP_F64 __llvm_amdgcn_ldexp_f64
#define BUILTIN_FLDEXP_F16 __llvm_amdgcn_ldexp_f16

#define BUILTIN_FREXP_EXP_F32 __llvm_amdgcn_frexp_exp_f32
#define BUILTIN_FREXP_EXP_F64 __llvm_amdgcn_frexp_exp_f64
#define BUILTIN_FREXP_EXP_F16 __llvm_amdgcn_frexp_exp_f16

#define BUILTIN_FREXP_MANT_F32 __llvm_amdgcn_frexp_mant_f32
#define BUILTIN_FREXP_MANT_F64 __llvm_amdgcn_frexp_mant_f64
#define BUILTIN_FREXP_MANT_F16 __llvm_amdgcn_frexp_mant_f16

#define BUILTIN_CMAX_F32 __llvm_maxnum_f32
#define BUILTIN_CMAX_F64 __llvm_maxnum_f64
#define BUILTIN_CMAX_F16 __llvm_maxnum_f16

#define BUILTIN_CMIN_F32 __llvm_minnum_f32
#define BUILTIN_CMIN_F64 __llvm_minnum_f64
#define BUILTIN_CMIN_F16 __llvm_minnum_f16

#define BUILTIN_TRIG_PREOP_F64 __llvm_amdgcn_trig_preop_f64

#define BUILTIN_MAX3_F32 __llvm_amdgcn_max3_f32
#define BUILTIN_MEDIAN3_F32 __llvm_amdgcn_med3_f32
#define BUILTIN_MIN3_F32 __llvm_amdgcn_min3_f32

#define BUILTIN_MAX3_S32 __llvm_amdgcn_max3_i32
#define BUILTIN_MEDIAN3_S32 __llvm_amdgcn_med3_i32
#define BUILTIN_MIN3_S32 __llvm_amdgcn_min3_i32

#define BUILTIN_MAX3_U32 __llvm_amdgcn_max3u_i32
#define BUILTIN_MEDIAN3_U32 __llvm_amdgcn_med3u_i32
#define BUILTIN_MIN3_U32 __llvm_amdgcn_min3u_i32

#define BUILTIN_MAD_F32 __llvm_fmuladd_f32
#define BUILTIN_MAD_F64 __llvm_fmuladd_f64
#define BUILTIN_MAD_F16 __llvm_fmuladd_f16

