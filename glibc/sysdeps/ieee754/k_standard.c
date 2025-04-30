/* @(#)k_standard.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: k_standard.c,v 1.6 1995/05/10 20:46:35 jtc Exp $";
#endif

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <errno.h>

#include <assert.h>

#if LIBM_SVID_COMPAT

# ifndef _USE_WRITE
#  include <stdio.h>			/* fputs(), stderr */
#  define	WRITE2(u,v)	fputs(u, stderr)
# else	/* !defined(_USE_WRITE) */
#  include <unistd.h>			/* write */
#  define	WRITE2(u,v)	write(2, u, v)
#  undef fflush
# endif	/* !defined(_USE_WRITE) */

/* XXX gcc versions until now don't delay the 0.0/0.0 division until
   runtime but produce NaN at compile time.  This is wrong since the
   exceptions are not set correctly.  */
# if 0
static const double zero = 0.0;	/* used as const */
# else
volatile static double zero = 0.0;	/* used as const */
# endif

/*
 * Standard conformance (non-IEEE) on exception cases.
 * Mapping:
 *	1 -- acos(|x|>1)
 *	2 -- asin(|x|>1)
 *	3 -- atan2(+-0,+-0)
 *	4 -- hypot overflow
 *	5 -- cosh overflow
 *	6 -- exp overflow
 *	7 -- exp underflow
 *	8 -- y0(0)
 *	9 -- y0(-ve)
 *	10-- y1(0)
 *	11-- y1(-ve)
 *	12-- yn(0)
 *	13-- yn(-ve)
 *	14-- lgamma(finite) overflow
 *	15-- lgamma(-integer)
 *	16-- log(0)
 *	17-- log(x<0)
 *	18-- log10(0)
 *	19-- log10(x<0)
 *	21-- pow(x,y) overflow
 *	22-- pow(x,y) underflow
 *	23-- pow(0,negative)
 *	24-- pow(neg,non-integral)
 *	25-- sinh(finite) overflow
 *	26-- sqrt(negative)
 *      27-- fmod(x,0)
 *      28-- remainder(x,0)
 *	29-- acosh(x<1)
 *	30-- atanh(|x|>1)
 *	31-- atanh(|x|=1)
 *	32-- scalb overflow
 *	33-- scalb underflow
 *	34-- j0(|x|>X_TLOSS)
 *	35-- y0(x>X_TLOSS)
 *	36-- j1(|x|>X_TLOSS)
 *	37-- y1(x>X_TLOSS)
 *	38-- jn(|x|>X_TLOSS, n)
 *	39-- yn(x>X_TLOSS, n)
 *	40-- tgamma(finite) overflow
 *	41-- tgamma(-integer)
 *	43-- +0**neg
 *	44-- exp2 overflow
 *	45-- exp2 underflow
 *	46-- exp10 overflow
 *	47-- exp10 underflow
 *	48-- log2(0)
 *	49-- log2(x<0)
 *	50-- tgamma(+-0)
 */


double
__kernel_standard(double x, double y, int type)
{
	struct exception exc;
# ifndef HUGE_VAL	/* this is the only routine that uses HUGE_VAL */
# define HUGE_VAL inf
	double inf = 0.0;

	SET_HIGH_WORD(inf,0x7ff00000);	/* set inf to infinite */
# endif

	/* The SVID struct exception uses a field "char *name;".  */
# define CSTR(func) ((char *) (type < 100				\
			      ? func					\
			      : (type < 200 ? func "f" : func "l")))

# ifdef _USE_WRITE
	(void) fflush(stdout);
# endif
	exc.arg1 = x;
	exc.arg2 = y;
	switch(type) {
	    case 1:
	    case 101:
	    case 201:
		/* acos(|x|>1) */
		exc.type = DOMAIN;
		exc.name = CSTR ("acos");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if(_LIB_VERSION == _SVID_) {
		    (void) WRITE2("acos: DOMAIN error\n", 19);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 2:
	    case 102:
	    case 202:
		/* asin(|x|>1) */
		exc.type = DOMAIN;
		exc.name = CSTR ("asin");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = NAN;
		if(_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if(_LIB_VERSION == _SVID_) {
			(void) WRITE2("asin: DOMAIN error\n", 19);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 3:
	    case 103:
	    case 203:
		/* atan2(+-0,+-0) */
		exc.arg1 = y;
		exc.arg2 = x;
		exc.type = DOMAIN;
		exc.name = CSTR ("atan2");
		assert (_LIB_VERSION == _SVID_);
		exc.retval = HUGE;
		if(_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if(_LIB_VERSION == _SVID_) {
			(void) WRITE2("atan2: DOMAIN error\n", 20);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 4:
	    case 104:
	    case 204:
		/* hypot(finite,finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("hypot");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 5:
	    case 105:
	    case 205:
		/* cosh(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("cosh");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 6:
	    case 106:
	    case 206:
		/* exp(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("exp");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 7:
	    case 107:
	    case 207:
		/* exp(finite) underflow */
		exc.type = UNDERFLOW;
		exc.name = CSTR ("exp");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 8:
	    case 108:
	    case 208:
		/* y0(0) = -inf */
		exc.type = DOMAIN;	/* should be SING for IEEE */
		exc.name = CSTR ("y0");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("y0: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 9:
	    case 109:
	    case 209:
		/* y0(x<0) = NaN */
		exc.type = DOMAIN;
		exc.name = CSTR ("y0");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("y0: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 10:
	    case 110:
	    case 210:
		/* y1(0) = -inf */
		exc.type = DOMAIN;	/* should be SING for IEEE */
		exc.name = CSTR ("y1");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("y1: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 11:
	    case 111:
	    case 211:
		/* y1(x<0) = NaN */
		exc.type = DOMAIN;
		exc.name = CSTR ("y1");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("y1: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 12:
	    case 112:
	    case 212:
		/* yn(n,0) = -inf */
		exc.type = DOMAIN;	/* should be SING for IEEE */
		exc.name = CSTR ("yn");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = ((x < 0 && ((int) x & 1) != 0)
				? HUGE_VAL
				: -HUGE_VAL);
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("yn: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 13:
	    case 113:
	    case 213:
		/* yn(x<0) = NaN */
		exc.type = DOMAIN;
		exc.name = CSTR ("yn");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("yn: DOMAIN error\n", 17);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 14:
	    case 114:
	    case 214:
		/* lgamma(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("lgamma");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 15:
	    case 115:
	    case 215:
		/* lgamma(-integer) or lgamma(0) */
		exc.type = SING;
		exc.name = CSTR ("lgamma");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("lgamma: SING error\n", 19);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 16:
	    case 116:
	    case 216:
		/* log(0) */
		exc.type = SING;
		exc.name = CSTR ("log");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log: SING error\n", 16);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 17:
	    case 117:
	    case 217:
		/* log(x<0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("log");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log: DOMAIN error\n", 18);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 18:
	    case 118:
	    case 218:
		/* log10(0) */
		exc.type = SING;
		exc.name = CSTR ("log10");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log10: SING error\n", 18);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 19:
	    case 119:
	    case 219:
		/* log10(x<0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("log10");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log10: DOMAIN error\n", 20);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 21:
	    case 121:
	    case 221:
		/* pow(x,y) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("pow");
		if (_LIB_VERSION == _SVID_) {
		  exc.retval = HUGE;
		  y *= 0.5;
		  if(x<zero&&rint(y)!=y) exc.retval = -HUGE;
		} else {
		  exc.retval = HUGE_VAL;
		  y *= 0.5;
		  if(x<zero&&rint(y)!=y) exc.retval = -HUGE_VAL;
		}
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 22:
	    case 122:
	    case 222:
		/* pow(x,y) underflow */
		exc.type = UNDERFLOW;
		exc.name = CSTR ("pow");
		exc.retval =  zero;
		y *= 0.5;
		if (x < zero && rint (y) != y)
		  exc.retval = -zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 23:
	    case 123:
	    case 223:
		/* -0**neg */
		exc.type = DOMAIN;
		exc.name = CSTR ("pow");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = zero;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("pow(0,neg): DOMAIN error\n", 25);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 43:
	    case 143:
	    case 243:
		/* +0**neg */
		exc.type = DOMAIN;
		exc.name = CSTR ("pow");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = zero;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("pow(0,neg): DOMAIN error\n", 25);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 24:
	    case 124:
	    case 224:
		/* neg**non-integral */
		exc.type = DOMAIN;
		exc.name = CSTR ("pow");
		if (_LIB_VERSION == _SVID_)
		    exc.retval = zero;
		else
		    exc.retval = zero/zero;	/* X/Open allow NaN */
		if (_LIB_VERSION == _POSIX_)
		   __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("neg**non-integral: DOMAIN error\n", 32);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 25:
	    case 125:
	    case 225:
		/* sinh(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("sinh");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = ( (x>zero) ? HUGE : -HUGE);
		else
		  exc.retval = ( (x>zero) ? HUGE_VAL : -HUGE_VAL);
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 26:
	    case 126:
	    case 226:
		/* sqrt(x<0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("sqrt");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = zero;
		else
		  exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("sqrt: DOMAIN error\n", 19);
		      }
		  __set_errno (EDOM);
		}
		break;
	    case 27:
	    case 127:
	    case 227:
		/* fmod(x,0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("fmod");
		if (_LIB_VERSION == _SVID_)
		    exc.retval = x;
		else
		    exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
		    (void) WRITE2("fmod:  DOMAIN error\n", 20);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 28:
	    case 128:
	    case 228:
		/* remainder(x,0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("remainder");
		exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
		    (void) WRITE2("remainder: DOMAIN error\n", 24);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 29:
	    case 129:
	    case 229:
		/* acosh(x<1) */
		exc.type = DOMAIN;
		exc.name = CSTR ("acosh");
		exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
		    (void) WRITE2("acosh: DOMAIN error\n", 20);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 30:
	    case 130:
	    case 230:
		/* atanh(|x|>1) */
		exc.type = DOMAIN;
		exc.name = CSTR ("atanh");
		exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
		    (void) WRITE2("atanh: DOMAIN error\n", 20);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 31:
	    case 131:
	    case 231:
		/* atanh(|x|=1) */
		exc.type = SING;
		exc.name = CSTR ("atanh");
		exc.retval = x/zero;	/* sign(x)*inf */
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
		    (void) WRITE2("atanh: SING error\n", 18);
		  }
		  __set_errno (EDOM);
		}
		break;
	    case 32:
	    case 132:
	    case 232:
		/* scalb overflow; SVID also returns +-HUGE_VAL */
		exc.type = OVERFLOW;
		exc.name = CSTR ("scalb");
		exc.retval = x > zero ? HUGE_VAL : -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 33:
	    case 133:
	    case 233:
		/* scalb underflow */
		exc.type = UNDERFLOW;
		exc.name = CSTR ("scalb");
		exc.retval = copysign(zero,x);
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 34:
	    case 134:
	    case 234:
		/* j0(|x|>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("j0");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 35:
	    case 135:
	    case 235:
		/* y0(x>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("y0");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 36:
	    case 136:
	    case 236:
		/* j1(|x|>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("j1");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 37:
	    case 137:
	    case 237:
		/* y1(x>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("y1");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 38:
	    case 138:
	    case 238:
		/* jn(|x|>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("jn");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 39:
	    case 139:
	    case 239:
		/* yn(x>X_TLOSS) */
		exc.type = TLOSS;
		exc.name = CSTR ("yn");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
			__set_errno (ERANGE);
		else if (!matherr(&exc)) {
			if (_LIB_VERSION == _SVID_) {
				(void) WRITE2(exc.name, 2);
				(void) WRITE2(": TLOSS error\n", 14);
			}
			__set_errno (ERANGE);
		}
		break;
	    case 40:
	    case 140:
	    case 240:
		/* tgamma(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("tgamma");
		exc.retval = copysign (HUGE_VAL, x);
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  __set_errno (ERANGE);
		}
		break;
	    case 41:
	    case 141:
	    case 241:
		/* tgamma(-integer) */
		exc.type = SING;
		exc.name = CSTR ("tgamma");
		exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("tgamma: SING error\n", 18);
			exc.retval = HUGE_VAL;
		      }
		  __set_errno (EDOM);
		}
		break;

	    case 44:
	    case 144:
	    case 244:
		/* exp(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("exp2");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 45:
	    case 145:
	    case 245:
		/* exp(finite) underflow */
		exc.type = UNDERFLOW;
		exc.name = CSTR ("exp2");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;

	    case 46:
	    case 146:
	    case 246:
		/* exp(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = CSTR ("exp10");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 47:
	    case 147:
	    case 247:
		/* exp(finite) underflow */
		exc.type = UNDERFLOW;
		exc.name = CSTR ("exp10");
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
			__set_errno (ERANGE);
		}
		break;
	    case 48:
	    case 148:
	    case 248:
		/* log2(0) */
		exc.type = SING;
		exc.name = CSTR ("log2");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  __set_errno (EDOM);
		}
		break;
	    case 49:
	    case 149:
	    case 249:
		/* log2(x<0) */
		exc.type = DOMAIN;
		exc.name = CSTR ("log2");
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = NAN;
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (EDOM);
		else if (!matherr(&exc)) {
		  __set_errno (EDOM);
		}
		break;
	    case 50:
	    case 150:
	    case 250:
		/* tgamma(+-0) */
		exc.type = SING;
		exc.name = CSTR ("tgamma");
		exc.retval = copysign (HUGE_VAL, x);
		if (_LIB_VERSION == _POSIX_)
		  __set_errno (ERANGE);
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_)
		    (void) WRITE2("tgamma: SING error\n", 18);
		  __set_errno (ERANGE);
		}
		break;

		/* #### Last used is 50/150/250 ### */

	    default:
		__builtin_unreachable ();
	}
	return exc.retval;
}
#endif
