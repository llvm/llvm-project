#include <fenv.h>
#include <fpu_control.h>

#define _FP_W_TYPE_SIZE		64
#define _FP_W_TYPE		unsigned long long
#define _FP_WS_TYPE		signed long long
#define _FP_I_TYPE		long long

#define _FP_MUL_MEAT_S(R,X,Y)					\
  _FP_MUL_MEAT_1_imm(_FP_WFRACBITS_S,R,X,Y)
#define _FP_MUL_MEAT_D(R,X,Y)					\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_Q(R,X,Y)					\
  _FP_MUL_MEAT_2_wide_3mul(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

#define _FP_DIV_MEAT_S(R,X,Y)	_FP_DIV_MEAT_1_imm(S,R,X,Y,_FP_DIV_HELP_imm)
#define _FP_DIV_MEAT_D(R,X,Y)	_FP_DIV_MEAT_1_udiv_norm(D,R,X,Y)
#define _FP_DIV_MEAT_Q(R,X,Y)	_FP_DIV_MEAT_2_udiv(Q,R,X,Y)

#define _FP_NANFRAC_S		((_FP_QNANBIT_S << 1) - 1)
#define _FP_NANFRAC_D		((_FP_QNANBIT_D << 1) - 1)
#define _FP_NANFRAC_Q		((_FP_QNANBIT_Q << 1) - 1), -1
#define _FP_NANSIGN_S		0
#define _FP_NANSIGN_D		0
#define _FP_NANSIGN_Q		0

#define _FP_KEEPNANFRACP 1
#define _FP_QNANNEGATEDP 0

/* From my experiments it seems X is chosen unless one of the
   NaNs is sNaN,  in which case the result is NANSIGN/NANFRAC.  */
#define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)			\
  do {								\
    if ((_FP_FRAC_HIGH_RAW_##fs(X)				\
	 | _FP_FRAC_HIGH_RAW_##fs(Y)) & _FP_QNANBIT_##fs)	\
      {								\
	R##_s = _FP_NANSIGN_##fs;				\
        _FP_FRAC_SET_##wc(R,_FP_NANFRAC_##fs);			\
      }								\
    else							\
      {								\
	R##_s = X##_s;						\
        _FP_FRAC_COPY_##wc(R,X);				\
      }								\
    R##_c = FP_CLS_NAN;						\
  } while (0)

#define _FP_DECL_EX		fpu_control_t _fcw

#define FP_ROUNDMODE		(_fcw & _FPU_FPCR_RM_MASK)

#define FP_RND_NEAREST		FE_TONEAREST
#define FP_RND_ZERO		FE_TOWARDZERO
#define FP_RND_PINF		FE_UPWARD
#define FP_RND_MINF		FE_DOWNWARD

#define FP_EX_INVALID		FE_INVALID
#define FP_EX_OVERFLOW		FE_OVERFLOW
#define FP_EX_UNDERFLOW		FE_UNDERFLOW
#define FP_EX_DIVZERO		FE_DIVBYZERO
#define FP_EX_INEXACT		FE_INEXACT

#define _FP_TININESS_AFTER_ROUNDING 0

#define FP_INIT_ROUNDMODE			\
do {						\
  _FPU_GETCW (_fcw);				\
} while (0)

#define FP_HANDLE_EXCEPTIONS						\
  do {									\
    const float fp_max = __FLT_MAX__;					\
    const float fp_min = __FLT_MIN__;					\
    const float fp_1e32 = 1.0e32f;					\
    const float fp_zero = 0.0;						\
    const float fp_one = 1.0;						\
    unsigned fpsr;							\
    if (_fex & FP_EX_INVALID)						\
      {									\
        __asm__ __volatile__ ("fdiv\ts0, %s0, %s0"			\
			      :						\
			      : "w" (fp_zero)				\
			      : "s0");					\
	__asm__ __volatile__ ("mrs\t%0, fpsr" : "=r" (fpsr));		\
      }									\
    if (_fex & FP_EX_DIVZERO)						\
      {									\
	__asm__ __volatile__ ("fdiv\ts0, %s0, %s1"			\
			      :						\
			      : "w" (fp_one), "w" (fp_zero)		\
			      : "s0");					\
	__asm__ __volatile__ ("mrs\t%0, fpsr" : "=r" (fpsr));		\
      }									\
    if (_fex & FP_EX_OVERFLOW)						\
      {									\
        __asm__ __volatile__ ("fadd\ts0, %s0, %s1"			\
			      :						\
			      : "w" (fp_max), "w" (fp_1e32)		\
			      : "s0");					\
        __asm__ __volatile__ ("mrs\t%0, fpsr" : "=r" (fpsr));		\
      }									\
    if (_fex & FP_EX_UNDERFLOW)						\
      {									\
	__asm__ __volatile__ ("fmul\ts0, %s0, %s0"			\
			      :						\
			      : "w" (fp_min)				\
			      : "s0");					\
	__asm__ __volatile__ ("mrs\t%0, fpsr" : "=r" (fpsr));		\
      }									\
    if (_fex & FP_EX_INEXACT)						\
      {									\
	__asm__ __volatile__ ("fsub\ts0, %s0, %s1"			\
			      :						\
			      : "w" (fp_max), "w" (fp_one)		\
			      : "s0");					\
	__asm__ __volatile__ ("mrs\t%0, fpsr" : "=r" (fpsr));		\
      }									\
  } while (0)

#define FP_TRAPPING_EXCEPTIONS ((_fcw >> FE_EXCEPT_SHIFT) & FE_ALL_EXCEPT)
