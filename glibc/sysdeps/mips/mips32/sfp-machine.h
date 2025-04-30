#define _FP_W_TYPE_SIZE		32
#define _FP_W_TYPE		unsigned long
#define _FP_WS_TYPE		signed long
#define _FP_I_TYPE		long

#define _FP_MUL_MEAT_S(R,X,Y)				\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_S,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_D(R,X,Y)				\
  _FP_MUL_MEAT_2_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_Q(R,X,Y)				\
  _FP_MUL_MEAT_4_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

#define _FP_MUL_MEAT_DW_S(R,X,Y)				\
  _FP_MUL_MEAT_DW_1_wide(_FP_WFRACBITS_S,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_DW_D(R,X,Y)				\
  _FP_MUL_MEAT_DW_2_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_DW_Q(R,X,Y)				\
  _FP_MUL_MEAT_DW_4_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

#define _FP_DIV_MEAT_S(R,X,Y)	_FP_DIV_MEAT_1_udiv_norm(S,R,X,Y)
#define _FP_DIV_MEAT_D(R,X,Y)	_FP_DIV_MEAT_2_udiv(D,R,X,Y)
#define _FP_DIV_MEAT_Q(R,X,Y)	_FP_DIV_MEAT_4_udiv(Q,R,X,Y)

#ifdef __mips_nan2008
# define _FP_NANFRAC_S		((_FP_QNANBIT_S << 1) - 1)
# define _FP_NANFRAC_D		((_FP_QNANBIT_D << 1) - 1), -1
# define _FP_NANFRAC_Q		((_FP_QNANBIT_Q << 1) - 1), -1, -1, -1
#else
# define _FP_NANFRAC_S		(_FP_QNANBIT_S - 1)
# define _FP_NANFRAC_D		(_FP_QNANBIT_D - 1), -1
# define _FP_NANFRAC_Q		(_FP_QNANBIT_Q - 1), -1, -1, -1
#endif
#define _FP_NANSIGN_S		0
#define _FP_NANSIGN_D		0
#define _FP_NANSIGN_Q		0

#define _FP_KEEPNANFRACP 1
#ifdef __mips_nan2008
# define _FP_QNANNEGATEDP 0
#else
# define _FP_QNANNEGATEDP 1
#endif

#ifdef __mips_nan2008
/* NaN payloads should be preserved for NAN2008.  */
# define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)	\
  do						\
    {						\
      R##_s = X##_s;				\
      _FP_FRAC_COPY_##wc (R, X);		\
      R##_c = FP_CLS_NAN;			\
    }						\
  while (0)
#else
/* From my experiments it seems X is chosen unless one of the
   NaNs is sNaN,  in which case the result is NANSIGN/NANFRAC.  */
# define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)			\
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
#endif

#define FP_EX_INVALID           (1 << 4)
#define FP_EX_DIVZERO           (1 << 3)
#define FP_EX_OVERFLOW          (1 << 2)
#define FP_EX_UNDERFLOW         (1 << 1)
#define FP_EX_INEXACT           (1 << 0)

#define _FP_TININESS_AFTER_ROUNDING 1
