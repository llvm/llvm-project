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

#include "irif.h"

#define BUILTIN_ABS_F32 __llvm_fabs_f32
#define BUILTIN_ABS_F64 __llvm_fabs_f64
#define BUILTIN_ABS_F16 __llvm_fabs_f16
#define BUILTIN_ABS_2F16 __llvm_fabs_2f16

#define BUILTIN_BITALIGN_B32 __llvm_amdgcn_alignbit

#define BUILTIN_CEIL_F32 __llvm_ceil_f32
#define BUILTIN_CEIL_F64 __llvm_ceil_f64
#define BUILTIN_CEIL_F16 __llvm_ceil_f16
#define BUILTIN_CEIL_2F16 __llvm_ceil_2f16

#define BUILTIN_CLASS_F32 __llvm_amdgcn_class_f32
#define BUILTIN_CLASS_F64 __llvm_amdgcn_class_f64
#define BUILTIN_CLASS_F16 __llvm_amdgcn_class_f16

#define BUILTIN_COPYSIGN_F32 __llvm_copysign_f32
#define BUILTIN_COPYSIGN_F64 __llvm_copysign_f64
#define BUILTIN_COPYSIGN_F16 __llvm_copysign_f16
#define BUILTIN_COPYSIGN_2F16 __llvm_copysign_2f16

#define BUILTIN_FIRSTBIT_U32(X) ((X) == 0 ? -1 : __builtin_clz(X))

#define BUILTIN_FLOOR_F32 __llvm_floor_f32
#define BUILTIN_FLOOR_F64 __llvm_floor_f64
#define BUILTIN_FLOOR_F16 __llvm_floor_f16
#define BUILTIN_FLOOR_2F16 __llvm_floor_2f16

#define BUILTIN_FRACTION_F32(X) ({ \
    float _fract_x = X; \
    float _fract_r = __llvm_amdgcn_fract_f32(_fract_x); \
    _fract_r = __llvm_amdgcn_class_f32(_fract_x, CLASS_PINF|CLASS_NINF) ? 0.0f : _fract_r; \
    _fract_r; \
})
#define BUILTIN_FRACTION_F64(X) ({ \
    double _fract_x = X; \
    double _fract_r = __llvm_amdgcn_fract_f64(_fract_x); \
    _fract_r = __llvm_amdgcn_class_f64(_fract_x, CLASS_PINF|CLASS_NINF) ? 0.0 : _fract_r; \
    _fract_r; \
})
#define BUILTIN_FRACTION_F16(X) ({ \
    half _fract_x = X; \
    half _fract_r = __llvm_amdgcn_fract_f16(_fract_x); \
    _fract_r = __llvm_amdgcn_class_f16(_fract_x, CLASS_PINF|CLASS_NINF) ? 0.0h : _fract_r; \
    _fract_r; \
})

#define BUILTIN_MAD_U32(A,B,C) ((A)*(B)+(C))

#define BUILTIN_MAX_F32 __llvm_maxnum_f32
#define BUILTIN_MAX_F64 __llvm_maxnum_f64
#define BUILTIN_MAX_F16 __llvm_maxnum_f16
#define BUILTIN_MAX_2F16 __llvm_maxnum_2f16

#define BUILTIN_MAX_S32(A,B) ((A) < (B) ? (B) : (A))
#define BUILTIN_MAX_U32(A,B) ((A) < (B) ? (B) : (A))

#define BUILTIN_MIN_F32 __llvm_minnum_f32
#define BUILTIN_MIN_F64 __llvm_minnum_f64
#define BUILTIN_MIN_F16 __llvm_minnum_f16
#define BUILTIN_MIN_2F16 __llvm_minnum_2f16

#define BUILTIN_MIN_S32(A,B) ((A) < (B) ? (A) : (B))
#define BUILTIN_MIN_U32(A,B) ((A) < (B) ? (A) : (B))

#define BUILTIN_CANONICALIZE_F32(X) __llvm_canonicalize_f32(X)
#define BUILTIN_CANONICALIZE_F64(X) __llvm_canonicalize_f64(X)
#define BUILTIN_CANONICALIZE_F16(X) __llvm_canonicalize_f16(X)
#define BUILTIN_CANONICALIZE_2F16(X) __llvm_canonicalize_2f16(X)

#define BUILTIN_MULHI_U32(A,B) (((ulong)(A) * (ulong)(B)) >> 32)

#define BUILTIN_COS_F32 __llvm_amdgcn_cos_f32

#define BUILTIN_EXP2_F32 __llvm_exp2_f32
#define BUILTIN_EXP2_F16 __llvm_exp2_f16

#define BUILTIN_LOG2_F32 __llvm_log2_f32
#define BUILTIN_LOG2_F16 __llvm_log2_f16

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
#define BUILTIN_RINT_2F16 __llvm_rint_2f16

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
#define BUILTIN_TRUNC_2F16 __llvm_trunc_2f16

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
#define BUILTIN_FMA_2F16 __llvm_fma_2f16

#define BUILTIN_FLDEXP_F32 __llvm_amdgcn_ldexp_f32
#define BUILTIN_FLDEXP_F64 __llvm_amdgcn_ldexp_f64
#define BUILTIN_FLDEXP_F16 __llvm_amdgcn_ldexp_f16

#define BUILTIN_FREXP_EXP_F32 __llvm_amdgcn_frexp_exp_i32_f32
#define BUILTIN_FREXP_EXP_F64 __llvm_amdgcn_frexp_exp_i32_f64
#define BUILTIN_FREXP_EXP_F16 __llvm_amdgcn_frexp_exp_i16_f16

#define BUILTIN_FREXP_MANT_F32 __llvm_amdgcn_frexp_mant_f32
#define BUILTIN_FREXP_MANT_F64 __llvm_amdgcn_frexp_mant_f64
#define BUILTIN_FREXP_MANT_F16 __llvm_amdgcn_frexp_mant_f16

#define BUILTIN_CMAX_F32 __llvm_maxnum_f32
#define BUILTIN_CMAX_F64 __llvm_maxnum_f64
#define BUILTIN_CMAX_F16 __llvm_maxnum_f16
#define BUILTIN_CMAX_2F16 __llvm_maxnum_2f16

#define BUILTIN_CMIN_F32 __llvm_minnum_f32
#define BUILTIN_CMIN_F64 __llvm_minnum_f64
#define BUILTIN_CMIN_F16 __llvm_minnum_f16
#define BUILTIN_CMIN_2F16 __llvm_minnum_2f16

#define BUILTIN_TRIG_PREOP_F64 __llvm_amdgcn_trig_preop_f64

#define BUILTIN_MAD_F32 __llvm_fmuladd_f32
#define BUILTIN_MAD_F64 __llvm_fmuladd_f64
#define BUILTIN_MAD_F16 __llvm_fmuladd_f16
#define BUILTIN_MAD_2F16 __llvm_fmuladd_2f16

// HW has ISA for max3, median3, and min3, median3 can be used to clamp
#define BUILTIN_CLAMP_S32(X,L,H) ({ \
    int _clamp_x = X; \
    int _clamp_l = L; \
    int _clamp_h = H; \
    int _clamp_r = _clamp_x > _clamp_l ? _clamp_x : _clamp_l; \
    _clamp_r = _clamp_r < _clamp_h ? _clamp_r : _clamp_h; \
    _clamp_r; \
})

#define BUILTIN_CLAMP_F32(X,L,H) __llvm_amdgcn_fmed3_f32(X,L,H)
#define BUILTIN_CLAMP_F16(X,L,H) __llvm_amdgcn_fmed3_f16(X,L,H)

