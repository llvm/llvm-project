/* Configure soft-fp for building sqrtf128.  Based on sfp-machine.h in
   libgcc, with soft-float and other irrelevant parts removed.  */

/* The type of the result of a floating point comparison.  This must
   match `__libgcc_cmp_return__' in GCC for the target.  */
#if defined (__clang__)
# ifdef __x86_64__
 typedef int __gcc_CMPtype __attribute__ ((mode (DI)));
# else
 typedef int __gcc_CMPtype __attribute__ ((mode (SI)));
# endif
#else
 typedef int __gcc_CMPtype __attribute__ ((mode (__libgcc_cmp_return__)));
#endif
#define CMPtype __gcc_CMPtype

#ifdef __x86_64__
# define _FP_W_TYPE_SIZE	64
# define _FP_W_TYPE		unsigned long long
# define _FP_WS_TYPE		signed long long
# define _FP_I_TYPE		long long

typedef int TItype __attribute__ ((mode (TI)));
typedef unsigned int UTItype __attribute__ ((mode (TI)));

# define TI_BITS (__CHAR_BIT__ * (int) sizeof (TItype))

# define _FP_MUL_MEAT_Q(R,X,Y)				\
  _FP_MUL_MEAT_2_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

# define _FP_DIV_MEAT_Q(R,X,Y)   _FP_DIV_MEAT_2_udiv(Q,R,X,Y)

# define _FP_NANFRAC_S		_FP_QNANBIT_S
# define _FP_NANFRAC_D		_FP_QNANBIT_D
# define _FP_NANFRAC_E		_FP_QNANBIT_E, 0
# define _FP_NANFRAC_Q		_FP_QNANBIT_Q, 0

# define FP_EX_SHIFT 7

# define _FP_DECL_EX \
  unsigned int _fcw __attribute__ ((unused)) = FP_RND_NEAREST;

# define FP_RND_NEAREST		0
# define FP_RND_ZERO		0x6000
# define FP_RND_PINF		0x4000
# define FP_RND_MINF		0x2000

# define FP_RND_MASK		0x6000

# define FP_INIT_ROUNDMODE					\
  do {								\
    __asm__ __volatile__ ("%vstmxcsr\t%0" : "=m" (_fcw));	\
  } while (0)
#else
# define _FP_W_TYPE_SIZE	32
# define _FP_W_TYPE		unsigned int
# define _FP_WS_TYPE		signed int
# define _FP_I_TYPE		int

# define __FP_FRAC_ADD_4(r3,r2,r1,r0,x3,x2,x1,x0,y3,y2,y1,y0)	\
  __asm__ ("add{l} {%11,%3|%3,%11}\n\t"				\
	   "adc{l} {%9,%2|%2,%9}\n\t"				\
	   "adc{l} {%7,%1|%1,%7}\n\t"				\
	   "adc{l} {%5,%0|%0,%5}"				\
	   : "=r" ((USItype) (r3)),				\
	     "=&r" ((USItype) (r2)),				\
	     "=&r" ((USItype) (r1)),				\
	     "=&r" ((USItype) (r0))				\
	   : "%0" ((USItype) (x3)),				\
	     "g" ((USItype) (y3)),				\
	     "%1" ((USItype) (x2)),				\
	     "g" ((USItype) (y2)),				\
	     "%2" ((USItype) (x1)),				\
	     "g" ((USItype) (y1)),				\
	     "%3" ((USItype) (x0)),				\
	     "g" ((USItype) (y0)))
# define __FP_FRAC_ADD_3(r2,r1,r0,x2,x1,x0,y2,y1,y0)		\
  __asm__ ("add{l} {%8,%2|%2,%8}\n\t"				\
	   "adc{l} {%6,%1|%1,%6}\n\t"				\
	   "adc{l} {%4,%0|%0,%4}"				\
	   : "=r" ((USItype) (r2)),				\
	     "=&r" ((USItype) (r1)),				\
	     "=&r" ((USItype) (r0))				\
	   : "%0" ((USItype) (x2)),				\
	     "g" ((USItype) (y2)),				\
	     "%1" ((USItype) (x1)),				\
	     "g" ((USItype) (y1)),				\
	     "%2" ((USItype) (x0)),				\
	     "g" ((USItype) (y0)))
# define __FP_FRAC_SUB_4(r3,r2,r1,r0,x3,x2,x1,x0,y3,y2,y1,y0)	\
  __asm__ ("sub{l} {%11,%3|%3,%11}\n\t"				\
	   "sbb{l} {%9,%2|%2,%9}\n\t"				\
	   "sbb{l} {%7,%1|%1,%7}\n\t"				\
	   "sbb{l} {%5,%0|%0,%5}"				\
	   : "=r" ((USItype) (r3)),				\
	     "=&r" ((USItype) (r2)),				\
	     "=&r" ((USItype) (r1)),				\
	     "=&r" ((USItype) (r0))				\
	   : "0" ((USItype) (x3)),				\
	     "g" ((USItype) (y3)),				\
	     "1" ((USItype) (x2)),				\
	     "g" ((USItype) (y2)),				\
	     "2" ((USItype) (x1)),				\
	     "g" ((USItype) (y1)),				\
	     "3" ((USItype) (x0)),				\
	     "g" ((USItype) (y0)))
# define __FP_FRAC_SUB_3(r2,r1,r0,x2,x1,x0,y2,y1,y0)		\
  __asm__ ("sub{l} {%8,%2|%2,%8}\n\t"				\
	   "sbb{l} {%6,%1|%1,%6}\n\t"				\
	   "sbb{l} {%4,%0|%0,%4}"				\
	   : "=r" ((USItype) (r2)),				\
	     "=&r" ((USItype) (r1)),				\
	     "=&r" ((USItype) (r0))				\
	   : "0" ((USItype) (x2)),				\
	     "g" ((USItype) (y2)),				\
	     "1" ((USItype) (x1)),				\
	     "g" ((USItype) (y1)),				\
	     "2" ((USItype) (x0)),				\
	     "g" ((USItype) (y0)))
# define __FP_FRAC_ADDI_4(x3,x2,x1,x0,i)			\
  __asm__ ("add{l} {%4,%3|%3,%4}\n\t"				\
	   "adc{l} {$0,%2|%2,0}\n\t"				\
	   "adc{l} {$0,%1|%1,0}\n\t"				\
	   "adc{l} {$0,%0|%0,0}"				\
	   : "+r" ((USItype) (x3)),				\
	     "+&r" ((USItype) (x2)),				\
	     "+&r" ((USItype) (x1)),				\
	     "+&r" ((USItype) (x0))				\
	   : "g" ((USItype) (i)))


# define _FP_MUL_MEAT_S(R,X,Y)				\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_S,R,X,Y,umul_ppmm)
# define _FP_MUL_MEAT_D(R,X,Y)				\
  _FP_MUL_MEAT_2_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
# define _FP_MUL_MEAT_Q(R,X,Y)				\
  _FP_MUL_MEAT_4_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

# define _FP_DIV_MEAT_S(R,X,Y)   _FP_DIV_MEAT_1_loop(S,R,X,Y)
# define _FP_DIV_MEAT_D(R,X,Y)   _FP_DIV_MEAT_2_udiv(D,R,X,Y)
# define _FP_DIV_MEAT_Q(R,X,Y)   _FP_DIV_MEAT_4_udiv(Q,R,X,Y)

# define _FP_NANFRAC_S		_FP_QNANBIT_S
# define _FP_NANFRAC_D		_FP_QNANBIT_D, 0
/* Even if XFmode is 12byte,  we have to pad it to
   16byte since soft-fp emulation is done in 16byte.  */
# define _FP_NANFRAC_E		_FP_QNANBIT_E, 0, 0, 0
# define _FP_NANFRAC_Q		_FP_QNANBIT_Q, 0, 0, 0

# define FP_EX_SHIFT 0

# define _FP_DECL_EX \
  unsigned short _fcw __attribute__ ((unused)) = FP_RND_NEAREST;

# define FP_RND_NEAREST		0
# define FP_RND_ZERO		0xc00
# define FP_RND_PINF		0x800
# define FP_RND_MINF		0x400

# define FP_RND_MASK		0xc00

# define FP_INIT_ROUNDMODE				\
  do {							\
    __asm__ __volatile__ ("fnstcw\t%0" : "=m" (_fcw));	\
  } while (0)
#endif

#define _FP_KEEPNANFRACP	1
#define _FP_QNANNEGATEDP 0

#define _FP_NANSIGN_S		1
#define _FP_NANSIGN_D		1
#define _FP_NANSIGN_E		1
#define _FP_NANSIGN_Q		1

/* Here is something Intel misdesigned: the specs don't define
   the case where we have two NaNs with same mantissas, but
   different sign. Different operations pick up different NaNs.  */
#define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)			\
  do {								\
    if (_FP_FRAC_GT_##wc(X, Y)					\
	|| (_FP_FRAC_EQ_##wc(X,Y) && (OP == '+' || OP == '*')))	\
      {								\
	R##_s = X##_s;						\
	_FP_FRAC_COPY_##wc(R,X);				\
      }								\
    else							\
      {								\
	R##_s = Y##_s;						\
	_FP_FRAC_COPY_##wc(R,Y);				\
      }								\
    R##_c = FP_CLS_NAN;						\
  } while (0)

#define FP_EX_INVALID		0x01
#define FP_EX_DENORM		0x02
#define FP_EX_DIVZERO		0x04
#define FP_EX_OVERFLOW		0x08
#define FP_EX_UNDERFLOW		0x10
#define FP_EX_INEXACT		0x20
#define FP_EX_ALL \
	(FP_EX_INVALID | FP_EX_DENORM | FP_EX_DIVZERO | FP_EX_OVERFLOW \
	 | FP_EX_UNDERFLOW | FP_EX_INEXACT)

void __sfp_handle_exceptions (int);

#define FP_HANDLE_EXCEPTIONS			\
  do {						\
    if (__builtin_expect (_fex, 0))		\
      __sfp_handle_exceptions (_fex);		\
  } while (0);

#define FP_TRAPPING_EXCEPTIONS ((~_fcw >> FP_EX_SHIFT) & FP_EX_ALL)

#define FP_ROUNDMODE		(_fcw & FP_RND_MASK)

#define _FP_TININESS_AFTER_ROUNDING 1
