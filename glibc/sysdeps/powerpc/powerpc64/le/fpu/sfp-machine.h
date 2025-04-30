#define _FP_W_TYPE_SIZE		64
#define _FP_W_TYPE		unsigned long long
#define _FP_WS_TYPE		signed long long
#define _FP_I_TYPE		long long

typedef int TItype __attribute__ ((mode (TI)));
typedef unsigned int UTItype __attribute__ ((mode (TI)));

#define TI_BITS (__CHAR_BIT__ * (int) sizeof (TItype))

/* The type of the result of a floating point comparison.  This must
   match `__libgcc_cmp_return__' in GCC for the target.  */
typedef int __gcc_CMPtype __attribute__ ((mode (__libgcc_cmp_return__)));
#define CMPtype __gcc_CMPtype

#define _FP_MUL_MEAT_S(R,X,Y)				\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_S,R,X,Y,umul_ppmm)

#define _FP_MUL_MEAT_D(R,X,Y)				\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_Q(R,X,Y)				\
  _FP_MUL_MEAT_2_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

#define _FP_DIV_MEAT_S(R,X,Y)	_FP_DIV_MEAT_1_loop(S,R,X,Y)

#define _FP_DIV_MEAT_D(R,X,Y)	_FP_DIV_MEAT_1_udiv(D,R,X,Y)
#define _FP_DIV_MEAT_Q(R,X,Y)   _FP_DIV_MEAT_2_udiv(Q,R,X,Y)

#define _FP_NANFRAC_S		((_FP_QNANBIT_S << 1) - 1)

#define _FP_NANFRAC_D		((_FP_QNANBIT_D << 1) - 1)
#define _FP_NANFRAC_Q		((_FP_QNANBIT_Q << 1) - 1), -1

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

#define	__LITTLE_ENDIAN	1234
#define	__BIG_ENDIAN	4321
#define	__BYTE_ORDER	__LITTLE_ENDIAN

/* Only provide exception support if we have hardware floating point using
   floating point registers and we can execute the mtfsf instruction.  This
   would only be true if we are using the emulation routines for IEEE 128-bit
   floating point on pre-ISA 3.0 machines without the IEEE 128-bit floating
   point support.  */

#ifdef __FLOAT128__
#define ISA_BIT(x) (1LL << (63 - x))

/* Use the same bits of the FPSCR.  */
# define FP_EX_INVALID		ISA_BIT(34)
# define FP_EX_OVERFLOW		ISA_BIT(35)
# define FP_EX_UNDERFLOW	ISA_BIT(36)
# define FP_EX_DIVZERO		ISA_BIT(37)
# define FP_EX_INEXACT		ISA_BIT(38)
# define FP_EX_ALL		(FP_EX_INVALID | FP_EX_OVERFLOW		\
				 | FP_EX_UNDERFLOW | FP_EX_DIVZERO	\
				 | FP_EX_INEXACT)

void __sfp_handle_exceptions (int);

# define FP_HANDLE_EXCEPTIONS			\
  do {						\
    if (__builtin_expect (_fex, 0))		\
      __sfp_handle_exceptions (_fex);		\
  } while (0);

/* The FP_EX_* bits track whether the exception has occurred.  This macro
   must set the FP_EX_* bits of those exceptions which are configured to
   trap.  The FPSCR bit which indicates this is 22 ISA bits above the
   respective FP_EX_* bit.  Note, the ISA labels bits from msb to lsb,
   so 22 ISA bits above is 22 bits below when counted from the lsb.  */
# define FP_TRAPPING_EXCEPTIONS ((_fpscr.i << 22) & FP_EX_ALL)

# define FP_RND_NEAREST	0x0
# define FP_RND_ZERO	0x1
# define FP_RND_PINF	0x2
# define FP_RND_MINF	0x3
# define FP_RND_MASK	0x3

# define _FP_DECL_EX \
  union { unsigned long long i; double d; } _fpscr __attribute__ ((unused)) = \
	{ .i = FP_RND_NEAREST }

#define FP_INIT_ROUNDMODE			\
  do {						\
    _fpscr.d = __builtin_mffs ();		\
  } while (0)

# define FP_ROUNDMODE	(_fpscr.i & FP_RND_MASK)
#endif	/* !__FLOAT128__ */
