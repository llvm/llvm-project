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

#define _FP_DIV_MEAT_S(R,X,Y)	_FP_DIV_MEAT_1_loop(S,R,X,Y)
#define _FP_DIV_MEAT_D(R,X,Y)	_FP_DIV_MEAT_2_udiv(D,R,X,Y)
#define _FP_DIV_MEAT_Q(R,X,Y)	_FP_DIV_MEAT_4_udiv(Q,R,X,Y)

#define _FP_NANFRAC_S		((_FP_QNANBIT_S << 1) - 1)
#define _FP_NANFRAC_D		((_FP_QNANBIT_D << 1) - 1), -1
#define _FP_NANFRAC_Q		((_FP_QNANBIT_Q << 1) - 1), -1, -1, -1
#define _FP_NANSIGN_S		0
#define _FP_NANSIGN_D		0
#define _FP_NANSIGN_Q		0

#define _FP_KEEPNANFRACP 1
#define _FP_QNANNEGATEDP 0

/* Someone please check this.  */
#define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)			\
  do {								\
    if ((_FP_FRAC_HIGH_RAW_##fs(X) & _FP_QNANBIT_##fs)		\
	&& !(_FP_FRAC_HIGH_RAW_##fs(Y) & _FP_QNANBIT_##fs))	\
      {								\
	R##_s = Y##_s;						\
	_FP_FRAC_COPY_##wc(R,Y);				\
      }								\
    else							\
      {								\
	R##_s = X##_s;						\
	_FP_FRAC_COPY_##wc(R,X);				\
      }								\
    R##_c = FP_CLS_NAN;						\
  } while (0)

#define _FP_TININESS_AFTER_ROUNDING 0

#if defined __NO_FPRS__ && !defined _SOFT_FLOAT

/* Exception flags.  We use the bit positions of the appropriate bits
   in the FPEFSCR.  */

# include <fenv_libc.h>
# include <sysdep.h>
# include <sys/prctl.h>

int __feraiseexcept_soft (int);
libc_hidden_proto (__feraiseexcept_soft)

# define FP_EX_INEXACT         SPEFSCR_FINXS
# define FP_EX_INVALID         SPEFSCR_FINVS
# define FP_EX_DIVZERO         SPEFSCR_FDBZS
# define FP_EX_UNDERFLOW       SPEFSCR_FUNFS
# define FP_EX_OVERFLOW        SPEFSCR_FOVFS

# define _FP_DECL_EX \
  int _spefscr __attribute__ ((unused)), _ftrapex __attribute__ ((unused)) = 0
# define FP_INIT_ROUNDMODE						\
  do									\
    {									\
      int _r;								\
									\
      _spefscr = fegetenv_register ();					\
      _r = INTERNAL_SYSCALL_CALL (prctl, PR_GET_FPEXC, &_ftrapex);	\
      if (INTERNAL_SYSCALL_ERROR_P (_r))				\
	_ftrapex = 0;							\
    }									\
  while (0)
# define FP_INIT_EXCEPTIONS /* Empty.  */

# define FP_HANDLE_EXCEPTIONS  __feraiseexcept_soft (_fex)
# define FP_ROUNDMODE          (_spefscr & 0x3)

/* Not correct in general, but sufficient for the uses in soft-fp.  */
# define FP_TRAPPING_EXCEPTIONS (_ftrapex & PR_FP_EXC_UND	\
				 ? FP_EX_UNDERFLOW		\
				 : 0)

#else

/* Exception flags.  We use the bit positions of the appropriate bits
   in the FPSCR, which also correspond to the FE_* bits.  This makes
   everything easier ;-).  */
# define FP_EX_INVALID         (1 << (31 - 2))
# define FP_EX_OVERFLOW        (1 << (31 - 3))
# define FP_EX_UNDERFLOW       (1 << (31 - 4))
# define FP_EX_DIVZERO         (1 << (31 - 5))
# define FP_EX_INEXACT         (1 << (31 - 6))

# define FP_HANDLE_EXCEPTIONS  __simulate_exceptions (_fex)
# define FP_ROUNDMODE          __sim_round_mode_thread
# define FP_TRAPPING_EXCEPTIONS \
  (~__sim_disabled_exceptions_thread & 0x3e000000)

#endif

extern __thread int __sim_exceptions_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_exceptions_thread, tls_model ("initial-exec"));
extern __thread int __sim_disabled_exceptions_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_disabled_exceptions_thread,
		       tls_model ("initial-exec"));
extern __thread int __sim_round_mode_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_round_mode_thread, tls_model ("initial-exec"));

extern void __simulate_exceptions (int x) attribute_hidden;
