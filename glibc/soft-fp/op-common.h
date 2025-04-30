/* Software floating-point emulation. Common operations.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com),
		  Jakub Jelinek (jj@ultra.linux.cz),
		  David S. Miller (davem@redhat.com) and
		  Peter Maydell (pmaydell@chiark.greenend.org.uk).

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   In addition to the permissions in the GNU Lesser General Public
   License, the Free Software Foundation gives you unlimited
   permission to link the compiled version of this file into
   combinations with other programs, and to distribute those
   combinations without any restriction coming from the use of this
   file.  (The Lesser General Public License restrictions do apply in
   other respects; for example, they cover modification of the file,
   and distribution when not linked into a combine executable.)

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef SOFT_FP_OP_COMMON_H
#define SOFT_FP_OP_COMMON_H	1

#define _FP_DECL(wc, X)						\
  _FP_I_TYPE X##_c __attribute__ ((unused)) _FP_ZERO_INIT;	\
  _FP_I_TYPE X##_s __attribute__ ((unused)) _FP_ZERO_INIT;	\
  _FP_I_TYPE X##_e __attribute__ ((unused)) _FP_ZERO_INIT;	\
  _FP_FRAC_DECL_##wc (X)

/* Test whether the qNaN bit denotes a signaling NaN.  */
#define _FP_FRAC_SNANP(fs, X)				\
  ((_FP_QNANNEGATEDP)					\
   ? (_FP_FRAC_HIGH_RAW_##fs (X) & _FP_QNANBIT_##fs)	\
   : !(_FP_FRAC_HIGH_RAW_##fs (X) & _FP_QNANBIT_##fs))
#define _FP_FRAC_SNANP_SEMIRAW(fs, X)			\
  ((_FP_QNANNEGATEDP)					\
   ? (_FP_FRAC_HIGH_##fs (X) & _FP_QNANBIT_SH_##fs)	\
   : !(_FP_FRAC_HIGH_##fs (X) & _FP_QNANBIT_SH_##fs))

/* Finish truly unpacking a native fp value by classifying the kind
   of fp value and normalizing both the exponent and the fraction.  */

#define _FP_UNPACK_CANONICAL(fs, wc, X)				\
  do								\
    {								\
      switch (X##_e)						\
	{							\
	default:						\
	  _FP_FRAC_HIGH_RAW_##fs (X) |= _FP_IMPLBIT_##fs;	\
	  _FP_FRAC_SLL_##wc (X, _FP_WORKBITS);			\
	  X##_e -= _FP_EXPBIAS_##fs;				\
	  X##_c = FP_CLS_NORMAL;				\
	  break;						\
								\
	case 0:							\
	  if (_FP_FRAC_ZEROP_##wc (X))				\
	    X##_c = FP_CLS_ZERO;				\
	  else if (FP_DENORM_ZERO)				\
	    {							\
	      X##_c = FP_CLS_ZERO;				\
	      _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);		\
	      FP_SET_EXCEPTION (FP_EX_DENORM);			\
	    }							\
	  else							\
	    {							\
	      /* A denormalized number.  */			\
	      _FP_I_TYPE _FP_UNPACK_CANONICAL_shift;		\
	      _FP_FRAC_CLZ_##wc (_FP_UNPACK_CANONICAL_shift,	\
				 X);				\
	      _FP_UNPACK_CANONICAL_shift -= _FP_FRACXBITS_##fs;	\
	      _FP_FRAC_SLL_##wc (X, (_FP_UNPACK_CANONICAL_shift \
				     + _FP_WORKBITS));		\
	      X##_e -= (_FP_EXPBIAS_##fs - 1			\
			+ _FP_UNPACK_CANONICAL_shift);		\
	      X##_c = FP_CLS_NORMAL;				\
	      FP_SET_EXCEPTION (FP_EX_DENORM);			\
	    }							\
	  break;						\
								\
	case _FP_EXPMAX_##fs:					\
	  if (_FP_FRAC_ZEROP_##wc (X))				\
	    X##_c = FP_CLS_INF;					\
	  else							\
	    {							\
	      X##_c = FP_CLS_NAN;				\
	      /* Check for signaling NaN.  */			\
	      if (_FP_FRAC_SNANP (fs, X))			\
		FP_SET_EXCEPTION (FP_EX_INVALID			\
				  | FP_EX_INVALID_SNAN);	\
	    }							\
	  break;						\
	}							\
    }								\
  while (0)

/* Finish unpacking an fp value in semi-raw mode: the mantissa is
   shifted by _FP_WORKBITS but the implicit MSB is not inserted and
   other classification is not done.  */
#define _FP_UNPACK_SEMIRAW(fs, wc, X)	_FP_FRAC_SLL_##wc (X, _FP_WORKBITS)

/* Check whether a raw or semi-raw input value should be flushed to
   zero, and flush it to zero if so.  */
#define _FP_CHECK_FLUSH_ZERO(fs, wc, X)			\
  do							\
    {							\
      if (FP_DENORM_ZERO				\
	  && X##_e == 0					\
	  && !_FP_FRAC_ZEROP_##wc (X))			\
	{						\
	  _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);	\
	  FP_SET_EXCEPTION (FP_EX_DENORM);		\
	}						\
    }							\
  while (0)

/* A semi-raw value has overflowed to infinity.  Adjust the mantissa
   and exponent appropriately.  */
#define _FP_OVERFLOW_SEMIRAW(fs, wc, X)			\
  do							\
    {							\
      if (FP_ROUNDMODE == FP_RND_NEAREST		\
	  || (FP_ROUNDMODE == FP_RND_PINF && !X##_s)	\
	  || (FP_ROUNDMODE == FP_RND_MINF && X##_s))	\
	{						\
	  X##_e = _FP_EXPMAX_##fs;			\
	  _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);	\
	}						\
      else						\
	{						\
	  X##_e = _FP_EXPMAX_##fs - 1;			\
	  _FP_FRAC_SET_##wc (X, _FP_MAXFRAC_##wc);	\
	}						\
      FP_SET_EXCEPTION (FP_EX_INEXACT);			\
      FP_SET_EXCEPTION (FP_EX_OVERFLOW);		\
    }							\
  while (0)

/* Check for a semi-raw value being a signaling NaN and raise the
   invalid exception if so.  */
#define _FP_CHECK_SIGNAN_SEMIRAW(fs, wc, X)			\
  do								\
    {								\
      if (X##_e == _FP_EXPMAX_##fs				\
	  && !_FP_FRAC_ZEROP_##wc (X)				\
	  && _FP_FRAC_SNANP_SEMIRAW (fs, X))			\
	FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_SNAN);	\
    }								\
  while (0)

/* Choose a NaN result from an operation on two semi-raw NaN
   values.  */
#define _FP_CHOOSENAN_SEMIRAW(fs, wc, R, X, Y, OP)			\
  do									\
    {									\
      /* _FP_CHOOSENAN expects raw values, so shift as required.  */	\
      _FP_FRAC_SRL_##wc (X, _FP_WORKBITS);				\
      _FP_FRAC_SRL_##wc (Y, _FP_WORKBITS);				\
      _FP_CHOOSENAN (fs, wc, R, X, Y, OP);				\
      _FP_FRAC_SLL_##wc (R, _FP_WORKBITS);				\
    }									\
  while (0)

/* Make the fractional part a quiet NaN, preserving the payload
   if possible, otherwise make it the canonical quiet NaN and set
   the sign bit accordingly.  */
#define _FP_SETQNAN(fs, wc, X)					\
  do								\
    {								\
      if (_FP_QNANNEGATEDP)					\
	{							\
	  _FP_FRAC_HIGH_RAW_##fs (X) &= _FP_QNANBIT_##fs - 1;	\
	  if (_FP_FRAC_ZEROP_##wc (X))				\
	    {							\
	      X##_s = _FP_NANSIGN_##fs;				\
	      _FP_FRAC_SET_##wc (X, _FP_NANFRAC_##fs);		\
	    }							\
	}							\
      else							\
	_FP_FRAC_HIGH_RAW_##fs (X) |= _FP_QNANBIT_##fs;		\
    }								\
  while (0)
#define _FP_SETQNAN_SEMIRAW(fs, wc, X)				\
  do								\
    {								\
      if (_FP_QNANNEGATEDP)					\
	{							\
	  _FP_FRAC_HIGH_##fs (X) &= _FP_QNANBIT_SH_##fs - 1;	\
	  if (_FP_FRAC_ZEROP_##wc (X))				\
	    {							\
	      X##_s = _FP_NANSIGN_##fs;				\
	      _FP_FRAC_SET_##wc (X, _FP_NANFRAC_##fs);		\
	      _FP_FRAC_SLL_##wc (X, _FP_WORKBITS);		\
	    }							\
	}							\
      else							\
	_FP_FRAC_HIGH_##fs (X) |= _FP_QNANBIT_SH_##fs;		\
    }								\
  while (0)

/* Test whether a biased exponent is normal (not zero or maximum).  */
#define _FP_EXP_NORMAL(fs, wc, X)	(((X##_e + 1) & _FP_EXPMAX_##fs) > 1)

/* Prepare to pack an fp value in semi-raw mode: the mantissa is
   rounded and shifted right, with the rounding possibly increasing
   the exponent (including changing a finite value to infinity).  */
#define _FP_PACK_SEMIRAW(fs, wc, X)				\
  do								\
    {								\
      int _FP_PACK_SEMIRAW_is_tiny				\
	= X##_e == 0 && !_FP_FRAC_ZEROP_##wc (X);		\
      if (_FP_TININESS_AFTER_ROUNDING				\
	  && _FP_PACK_SEMIRAW_is_tiny)				\
	{							\
	  FP_DECL_##fs (_FP_PACK_SEMIRAW_T);			\
	  _FP_FRAC_COPY_##wc (_FP_PACK_SEMIRAW_T, X);		\
	  _FP_PACK_SEMIRAW_T##_s = X##_s;			\
	  _FP_PACK_SEMIRAW_T##_e = X##_e;			\
	  _FP_FRAC_SLL_##wc (_FP_PACK_SEMIRAW_T, 1);		\
	  _FP_ROUND (wc, _FP_PACK_SEMIRAW_T);			\
	  if (_FP_FRAC_OVERP_##wc (fs, _FP_PACK_SEMIRAW_T))	\
	    _FP_PACK_SEMIRAW_is_tiny = 0;			\
	}							\
      _FP_ROUND (wc, X);					\
      if (_FP_PACK_SEMIRAW_is_tiny)				\
	{							\
	  if ((FP_CUR_EXCEPTIONS & FP_EX_INEXACT)		\
	      || (FP_TRAPPING_EXCEPTIONS & FP_EX_UNDERFLOW))	\
	    FP_SET_EXCEPTION (FP_EX_UNDERFLOW);			\
	}							\
      if (_FP_FRAC_HIGH_##fs (X)				\
	  & (_FP_OVERFLOW_##fs >> 1))				\
	{							\
	  _FP_FRAC_HIGH_##fs (X) &= ~(_FP_OVERFLOW_##fs >> 1);	\
	  X##_e++;						\
	  if (X##_e == _FP_EXPMAX_##fs)				\
	    _FP_OVERFLOW_SEMIRAW (fs, wc, X);			\
	}							\
      _FP_FRAC_SRL_##wc (X, _FP_WORKBITS);			\
      if (X##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (X))	\
	{							\
	  if (!_FP_KEEPNANFRACP)				\
	    {							\
	      _FP_FRAC_SET_##wc (X, _FP_NANFRAC_##fs);		\
	      X##_s = _FP_NANSIGN_##fs;				\
	    }							\
	  else							\
	    _FP_SETQNAN (fs, wc, X);				\
	}							\
    }								\
  while (0)

/* Before packing the bits back into the native fp result, take care
   of such mundane things as rounding and overflow.  Also, for some
   kinds of fp values, the original parts may not have been fully
   extracted -- but that is ok, we can regenerate them now.  */

#define _FP_PACK_CANONICAL(fs, wc, X)					\
  do									\
    {									\
      switch (X##_c)							\
	{								\
	case FP_CLS_NORMAL:						\
	  X##_e += _FP_EXPBIAS_##fs;					\
	  if (X##_e > 0)						\
	    {								\
	      _FP_ROUND (wc, X);					\
	      if (_FP_FRAC_OVERP_##wc (fs, X))				\
		{							\
		  _FP_FRAC_CLEAR_OVERP_##wc (fs, X);			\
		  X##_e++;						\
		}							\
	      _FP_FRAC_SRL_##wc (X, _FP_WORKBITS);			\
	      if (X##_e >= _FP_EXPMAX_##fs)				\
		{							\
		  /* Overflow.  */					\
		  switch (FP_ROUNDMODE)					\
		    {							\
		    case FP_RND_NEAREST:				\
		      X##_c = FP_CLS_INF;				\
		      break;						\
		    case FP_RND_PINF:					\
		      if (!X##_s)					\
			X##_c = FP_CLS_INF;				\
		      break;						\
		    case FP_RND_MINF:					\
		      if (X##_s)					\
			X##_c = FP_CLS_INF;				\
		      break;						\
		    }							\
		  if (X##_c == FP_CLS_INF)				\
		    {							\
		      /* Overflow to infinity.  */			\
		      X##_e = _FP_EXPMAX_##fs;				\
		      _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);		\
		    }							\
		  else							\
		    {							\
		      /* Overflow to maximum normal.  */		\
		      X##_e = _FP_EXPMAX_##fs - 1;			\
		      _FP_FRAC_SET_##wc (X, _FP_MAXFRAC_##wc);		\
		    }							\
		  FP_SET_EXCEPTION (FP_EX_OVERFLOW);			\
		  FP_SET_EXCEPTION (FP_EX_INEXACT);			\
		}							\
	    }								\
	  else								\
	    {								\
	      /* We've got a denormalized number.  */			\
	      int _FP_PACK_CANONICAL_is_tiny = 1;			\
	      if (_FP_TININESS_AFTER_ROUNDING && X##_e == 0)		\
		{							\
		  FP_DECL_##fs (_FP_PACK_CANONICAL_T);			\
		  _FP_FRAC_COPY_##wc (_FP_PACK_CANONICAL_T, X);		\
		  _FP_PACK_CANONICAL_T##_s = X##_s;			\
		  _FP_PACK_CANONICAL_T##_e = X##_e;			\
		  _FP_ROUND (wc, _FP_PACK_CANONICAL_T);			\
		  if (_FP_FRAC_OVERP_##wc (fs, _FP_PACK_CANONICAL_T))	\
		    _FP_PACK_CANONICAL_is_tiny = 0;			\
		}							\
	      X##_e = -X##_e + 1;					\
	      if (X##_e <= _FP_WFRACBITS_##fs)				\
		{							\
		  _FP_FRAC_SRS_##wc (X, X##_e, _FP_WFRACBITS_##fs);	\
		  _FP_ROUND (wc, X);					\
		  if (_FP_FRAC_HIGH_##fs (X)				\
		      & (_FP_OVERFLOW_##fs >> 1))			\
		    {							\
		      X##_e = 1;					\
		      _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);		\
		      FP_SET_EXCEPTION (FP_EX_INEXACT);			\
		    }							\
		  else							\
		    {							\
		      X##_e = 0;					\
		      _FP_FRAC_SRL_##wc (X, _FP_WORKBITS);		\
		    }							\
		  if (_FP_PACK_CANONICAL_is_tiny			\
		      && ((FP_CUR_EXCEPTIONS & FP_EX_INEXACT)		\
			  || (FP_TRAPPING_EXCEPTIONS			\
			      & FP_EX_UNDERFLOW)))			\
		    FP_SET_EXCEPTION (FP_EX_UNDERFLOW);			\
		}							\
	      else							\
		{							\
		  /* Underflow to zero.  */				\
		  X##_e = 0;						\
		  if (!_FP_FRAC_ZEROP_##wc (X))				\
		    {							\
		      _FP_FRAC_SET_##wc (X, _FP_MINFRAC_##wc);		\
		      _FP_ROUND (wc, X);				\
		      _FP_FRAC_LOW_##wc (X) >>= (_FP_WORKBITS);		\
		    }							\
		  FP_SET_EXCEPTION (FP_EX_UNDERFLOW);			\
		}							\
	    }								\
	  break;							\
									\
	case FP_CLS_ZERO:						\
	  X##_e = 0;							\
	  _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);			\
	  break;							\
									\
	case FP_CLS_INF:						\
	  X##_e = _FP_EXPMAX_##fs;					\
	  _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);			\
	  break;							\
									\
	case FP_CLS_NAN:						\
	  X##_e = _FP_EXPMAX_##fs;					\
	  if (!_FP_KEEPNANFRACP)					\
	    {								\
	      _FP_FRAC_SET_##wc (X, _FP_NANFRAC_##fs);			\
	      X##_s = _FP_NANSIGN_##fs;					\
	    }								\
	  else								\
	    _FP_SETQNAN (fs, wc, X);					\
	  break;							\
	}								\
    }									\
  while (0)

/* This one accepts raw argument and not cooked,  returns
   1 if X is a signaling NaN.  */
#define _FP_ISSIGNAN(fs, wc, X)			\
  ({						\
    int _FP_ISSIGNAN_ret = 0;			\
    if (X##_e == _FP_EXPMAX_##fs)		\
      {						\
	if (!_FP_FRAC_ZEROP_##wc (X)		\
	    && _FP_FRAC_SNANP (fs, X))		\
	  _FP_ISSIGNAN_ret = 1;			\
      }						\
    _FP_ISSIGNAN_ret;				\
  })





/* Addition on semi-raw values.  */
#define _FP_ADD_INTERNAL(fs, wc, R, X, Y, OP)				\
  do									\
    {									\
      _FP_CHECK_FLUSH_ZERO (fs, wc, X);					\
      _FP_CHECK_FLUSH_ZERO (fs, wc, Y);					\
      if (X##_s == Y##_s)						\
	{								\
	  /* Addition.  */						\
	  __label__ add1, add2, add3, add_done;				\
	  R##_s = X##_s;						\
	  int _FP_ADD_INTERNAL_ediff = X##_e - Y##_e;			\
	  if (_FP_ADD_INTERNAL_ediff > 0)				\
	    {								\
	      R##_e = X##_e;						\
	      if (Y##_e == 0)						\
		{							\
		  /* Y is zero or denormalized.  */			\
		  if (_FP_FRAC_ZEROP_##wc (Y))				\
		    {							\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
		      _FP_FRAC_COPY_##wc (R, X);			\
		      goto add_done;					\
		    }							\
		  else							\
		    {							\
		      FP_SET_EXCEPTION (FP_EX_DENORM);			\
		      _FP_ADD_INTERNAL_ediff--;				\
		      if (_FP_ADD_INTERNAL_ediff == 0)			\
			{						\
			  _FP_FRAC_ADD_##wc (R, X, Y);			\
			  goto add3;					\
			}						\
		      if (X##_e == _FP_EXPMAX_##fs)			\
			{						\
			  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
			  _FP_FRAC_COPY_##wc (R, X);			\
			  goto add_done;				\
			}						\
		      goto add1;					\
		    }							\
		}							\
	      else if (X##_e == _FP_EXPMAX_##fs)			\
		{							\
		  /* X is NaN or Inf, Y is normal.  */			\
		  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);			\
		  _FP_FRAC_COPY_##wc (R, X);				\
		  goto add_done;					\
		}							\
									\
	      /* Insert implicit MSB of Y.  */				\
	      _FP_FRAC_HIGH_##fs (Y) |= _FP_IMPLBIT_SH_##fs;		\
									\
	    add1:							\
	      /* Shift the mantissa of Y to the right			\
		 _FP_ADD_INTERNAL_EDIFF steps; remember to account	\
		 later for the implicit MSB of X.  */			\
	      if (_FP_ADD_INTERNAL_ediff <= _FP_WFRACBITS_##fs)		\
		_FP_FRAC_SRS_##wc (Y, _FP_ADD_INTERNAL_ediff,		\
				   _FP_WFRACBITS_##fs);			\
	      else if (!_FP_FRAC_ZEROP_##wc (Y))			\
		_FP_FRAC_SET_##wc (Y, _FP_MINFRAC_##wc);		\
	      _FP_FRAC_ADD_##wc (R, X, Y);				\
	    }								\
	  else if (_FP_ADD_INTERNAL_ediff < 0)				\
	    {								\
	      _FP_ADD_INTERNAL_ediff = -_FP_ADD_INTERNAL_ediff;		\
	      R##_e = Y##_e;						\
	      if (X##_e == 0)						\
		{							\
		  /* X is zero or denormalized.  */			\
		  if (_FP_FRAC_ZEROP_##wc (X))				\
		    {							\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
		      _FP_FRAC_COPY_##wc (R, Y);			\
		      goto add_done;					\
		    }							\
		  else							\
		    {							\
		      FP_SET_EXCEPTION (FP_EX_DENORM);			\
		      _FP_ADD_INTERNAL_ediff--;				\
		      if (_FP_ADD_INTERNAL_ediff == 0)			\
			{						\
			  _FP_FRAC_ADD_##wc (R, Y, X);			\
			  goto add3;					\
			}						\
		      if (Y##_e == _FP_EXPMAX_##fs)			\
			{						\
			  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
			  _FP_FRAC_COPY_##wc (R, Y);			\
			  goto add_done;				\
			}						\
		      goto add2;					\
		    }							\
		}							\
	      else if (Y##_e == _FP_EXPMAX_##fs)			\
		{							\
		  /* Y is NaN or Inf, X is normal.  */			\
		  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);			\
		  _FP_FRAC_COPY_##wc (R, Y);				\
		  goto add_done;					\
		}							\
									\
	      /* Insert implicit MSB of X.  */				\
	      _FP_FRAC_HIGH_##fs (X) |= _FP_IMPLBIT_SH_##fs;		\
									\
	    add2:							\
	      /* Shift the mantissa of X to the right			\
		 _FP_ADD_INTERNAL_EDIFF steps; remember to account	\
		 later for the implicit MSB of Y.  */			\
	      if (_FP_ADD_INTERNAL_ediff <= _FP_WFRACBITS_##fs)		\
		_FP_FRAC_SRS_##wc (X, _FP_ADD_INTERNAL_ediff,		\
				   _FP_WFRACBITS_##fs);			\
	      else if (!_FP_FRAC_ZEROP_##wc (X))			\
		_FP_FRAC_SET_##wc (X, _FP_MINFRAC_##wc);		\
	      _FP_FRAC_ADD_##wc (R, Y, X);				\
	    }								\
	  else								\
	    {								\
	      /* _FP_ADD_INTERNAL_ediff == 0.  */			\
	      if (!_FP_EXP_NORMAL (fs, wc, X))				\
		{							\
		  if (X##_e == 0)					\
		    {							\
		      /* X and Y are zero or denormalized.  */		\
		      R##_e = 0;					\
		      if (_FP_FRAC_ZEROP_##wc (X))			\
			{						\
			  if (!_FP_FRAC_ZEROP_##wc (Y))			\
			    FP_SET_EXCEPTION (FP_EX_DENORM);		\
			  _FP_FRAC_COPY_##wc (R, Y);			\
			  goto add_done;				\
			}						\
		      else if (_FP_FRAC_ZEROP_##wc (Y))			\
			{						\
			  FP_SET_EXCEPTION (FP_EX_DENORM);		\
			  _FP_FRAC_COPY_##wc (R, X);			\
			  goto add_done;				\
			}						\
		      else						\
			{						\
			  FP_SET_EXCEPTION (FP_EX_DENORM);		\
			  _FP_FRAC_ADD_##wc (R, X, Y);			\
			  if (_FP_FRAC_HIGH_##fs (R) & _FP_IMPLBIT_SH_##fs) \
			    {						\
			      /* Normalized result.  */			\
			      _FP_FRAC_HIGH_##fs (R)			\
				&= ~(_FP_W_TYPE) _FP_IMPLBIT_SH_##fs;	\
			      R##_e = 1;				\
			    }						\
			  goto add_done;				\
			}						\
		    }							\
		  else							\
		    {							\
		      /* X and Y are NaN or Inf.  */			\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
		      R##_e = _FP_EXPMAX_##fs;				\
		      if (_FP_FRAC_ZEROP_##wc (X))			\
			_FP_FRAC_COPY_##wc (R, Y);			\
		      else if (_FP_FRAC_ZEROP_##wc (Y))			\
			_FP_FRAC_COPY_##wc (R, X);			\
		      else						\
			_FP_CHOOSENAN_SEMIRAW (fs, wc, R, X, Y, OP);	\
		      goto add_done;					\
		    }							\
		}							\
	      /* The exponents of X and Y, both normal, are equal.  The	\
		 implicit MSBs will always add to increase the		\
		 exponent.  */						\
	      _FP_FRAC_ADD_##wc (R, X, Y);				\
	      R##_e = X##_e + 1;					\
	      _FP_FRAC_SRS_##wc (R, 1, _FP_WFRACBITS_##fs);		\
	      if (R##_e == _FP_EXPMAX_##fs)				\
		/* Overflow to infinity (depending on rounding mode).  */ \
		_FP_OVERFLOW_SEMIRAW (fs, wc, R);			\
	      goto add_done;						\
	    }								\
	add3:								\
	  if (_FP_FRAC_HIGH_##fs (R) & _FP_IMPLBIT_SH_##fs)		\
	    {								\
	      /* Overflow.  */						\
	      _FP_FRAC_HIGH_##fs (R) &= ~(_FP_W_TYPE) _FP_IMPLBIT_SH_##fs; \
	      R##_e++;							\
	      _FP_FRAC_SRS_##wc (R, 1, _FP_WFRACBITS_##fs);		\
	      if (R##_e == _FP_EXPMAX_##fs)				\
		/* Overflow to infinity (depending on rounding mode).  */ \
		_FP_OVERFLOW_SEMIRAW (fs, wc, R);			\
	    }								\
	add_done: ;							\
	}								\
      else								\
	{								\
	  /* Subtraction.  */						\
	  __label__ sub1, sub2, sub3, norm, sub_done;			\
	  int _FP_ADD_INTERNAL_ediff = X##_e - Y##_e;			\
	  if (_FP_ADD_INTERNAL_ediff > 0)				\
	    {								\
	      R##_e = X##_e;						\
	      R##_s = X##_s;						\
	      if (Y##_e == 0)						\
		{							\
		  /* Y is zero or denormalized.  */			\
		  if (_FP_FRAC_ZEROP_##wc (Y))				\
		    {							\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
		      _FP_FRAC_COPY_##wc (R, X);			\
		      goto sub_done;					\
		    }							\
		  else							\
		    {							\
		      FP_SET_EXCEPTION (FP_EX_DENORM);			\
		      _FP_ADD_INTERNAL_ediff--;				\
		      if (_FP_ADD_INTERNAL_ediff == 0)			\
			{						\
			  _FP_FRAC_SUB_##wc (R, X, Y);			\
			  goto sub3;					\
			}						\
		      if (X##_e == _FP_EXPMAX_##fs)			\
			{						\
			  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
			  _FP_FRAC_COPY_##wc (R, X);			\
			  goto sub_done;				\
			}						\
		      goto sub1;					\
		    }							\
		}							\
	      else if (X##_e == _FP_EXPMAX_##fs)			\
		{							\
		  /* X is NaN or Inf, Y is normal.  */			\
		  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);			\
		  _FP_FRAC_COPY_##wc (R, X);				\
		  goto sub_done;					\
		}							\
									\
	      /* Insert implicit MSB of Y.  */				\
	      _FP_FRAC_HIGH_##fs (Y) |= _FP_IMPLBIT_SH_##fs;		\
									\
	    sub1:							\
	      /* Shift the mantissa of Y to the right			\
		 _FP_ADD_INTERNAL_EDIFF steps; remember to account	\
		 later for the implicit MSB of X.  */			\
	      if (_FP_ADD_INTERNAL_ediff <= _FP_WFRACBITS_##fs)		\
		_FP_FRAC_SRS_##wc (Y, _FP_ADD_INTERNAL_ediff,		\
				   _FP_WFRACBITS_##fs);			\
	      else if (!_FP_FRAC_ZEROP_##wc (Y))			\
		_FP_FRAC_SET_##wc (Y, _FP_MINFRAC_##wc);		\
	      _FP_FRAC_SUB_##wc (R, X, Y);				\
	    }								\
	  else if (_FP_ADD_INTERNAL_ediff < 0)				\
	    {								\
	      _FP_ADD_INTERNAL_ediff = -_FP_ADD_INTERNAL_ediff;		\
	      R##_e = Y##_e;						\
	      R##_s = Y##_s;						\
	      if (X##_e == 0)						\
		{							\
		  /* X is zero or denormalized.  */			\
		  if (_FP_FRAC_ZEROP_##wc (X))				\
		    {							\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
		      _FP_FRAC_COPY_##wc (R, Y);			\
		      goto sub_done;					\
		    }							\
		  else							\
		    {							\
		      FP_SET_EXCEPTION (FP_EX_DENORM);			\
		      _FP_ADD_INTERNAL_ediff--;				\
		      if (_FP_ADD_INTERNAL_ediff == 0)			\
			{						\
			  _FP_FRAC_SUB_##wc (R, Y, X);			\
			  goto sub3;					\
			}						\
		      if (Y##_e == _FP_EXPMAX_##fs)			\
			{						\
			  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
			  _FP_FRAC_COPY_##wc (R, Y);			\
			  goto sub_done;				\
			}						\
		      goto sub2;					\
		    }							\
		}							\
	      else if (Y##_e == _FP_EXPMAX_##fs)			\
		{							\
		  /* Y is NaN or Inf, X is normal.  */			\
		  _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);			\
		  _FP_FRAC_COPY_##wc (R, Y);				\
		  goto sub_done;					\
		}							\
									\
	      /* Insert implicit MSB of X.  */				\
	      _FP_FRAC_HIGH_##fs (X) |= _FP_IMPLBIT_SH_##fs;		\
									\
	    sub2:							\
	      /* Shift the mantissa of X to the right			\
		 _FP_ADD_INTERNAL_EDIFF steps; remember to account	\
		 later for the implicit MSB of Y.  */			\
	      if (_FP_ADD_INTERNAL_ediff <= _FP_WFRACBITS_##fs)		\
		_FP_FRAC_SRS_##wc (X, _FP_ADD_INTERNAL_ediff,		\
				   _FP_WFRACBITS_##fs);			\
	      else if (!_FP_FRAC_ZEROP_##wc (X))			\
		_FP_FRAC_SET_##wc (X, _FP_MINFRAC_##wc);		\
	      _FP_FRAC_SUB_##wc (R, Y, X);				\
	    }								\
	  else								\
	    {								\
	      /* ediff == 0.  */					\
	      if (!_FP_EXP_NORMAL (fs, wc, X))				\
		{							\
		  if (X##_e == 0)					\
		    {							\
		      /* X and Y are zero or denormalized.  */		\
		      R##_e = 0;					\
		      if (_FP_FRAC_ZEROP_##wc (X))			\
			{						\
			  _FP_FRAC_COPY_##wc (R, Y);			\
			  if (_FP_FRAC_ZEROP_##wc (Y))			\
			    R##_s = (FP_ROUNDMODE == FP_RND_MINF);	\
			  else						\
			    {						\
			      FP_SET_EXCEPTION (FP_EX_DENORM);		\
			      R##_s = Y##_s;				\
			    }						\
			  goto sub_done;				\
			}						\
		      else if (_FP_FRAC_ZEROP_##wc (Y))			\
			{						\
			  FP_SET_EXCEPTION (FP_EX_DENORM);		\
			  _FP_FRAC_COPY_##wc (R, X);			\
			  R##_s = X##_s;				\
			  goto sub_done;				\
			}						\
		      else						\
			{						\
			  FP_SET_EXCEPTION (FP_EX_DENORM);		\
			  _FP_FRAC_SUB_##wc (R, X, Y);			\
			  R##_s = X##_s;				\
			  if (_FP_FRAC_HIGH_##fs (R) & _FP_IMPLBIT_SH_##fs) \
			    {						\
			      /* |X| < |Y|, negate result.  */		\
			      _FP_FRAC_SUB_##wc (R, Y, X);		\
			      R##_s = Y##_s;				\
			    }						\
			  else if (_FP_FRAC_ZEROP_##wc (R))		\
			    R##_s = (FP_ROUNDMODE == FP_RND_MINF);	\
			  goto sub_done;				\
			}						\
		    }							\
		  else							\
		    {							\
		      /* X and Y are NaN or Inf, of opposite signs.  */	\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, X);		\
		      _FP_CHECK_SIGNAN_SEMIRAW (fs, wc, Y);		\
		      R##_e = _FP_EXPMAX_##fs;				\
		      if (_FP_FRAC_ZEROP_##wc (X))			\
			{						\
			  if (_FP_FRAC_ZEROP_##wc (Y))			\
			    {						\
			      /* Inf - Inf.  */				\
			      R##_s = _FP_NANSIGN_##fs;			\
			      _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);	\
			      _FP_FRAC_SLL_##wc (R, _FP_WORKBITS);	\
			      FP_SET_EXCEPTION (FP_EX_INVALID		\
						| FP_EX_INVALID_ISI);	\
			    }						\
			  else						\
			    {						\
			      /* Inf - NaN.  */				\
			      R##_s = Y##_s;				\
			      _FP_FRAC_COPY_##wc (R, Y);		\
			    }						\
			}						\
		      else						\
			{						\
			  if (_FP_FRAC_ZEROP_##wc (Y))			\
			    {						\
			      /* NaN - Inf.  */				\
			      R##_s = X##_s;				\
			      _FP_FRAC_COPY_##wc (R, X);		\
			    }						\
			  else						\
			    {						\
			      /* NaN - NaN.  */				\
			      _FP_CHOOSENAN_SEMIRAW (fs, wc, R, X, Y, OP); \
			    }						\
			}						\
		      goto sub_done;					\
		    }							\
		}							\
	      /* The exponents of X and Y, both normal, are equal.  The	\
		 implicit MSBs cancel.  */				\
	      R##_e = X##_e;						\
	      _FP_FRAC_SUB_##wc (R, X, Y);				\
	      R##_s = X##_s;						\
	      if (_FP_FRAC_HIGH_##fs (R) & _FP_IMPLBIT_SH_##fs)		\
		{							\
		  /* |X| < |Y|, negate result.  */			\
		  _FP_FRAC_SUB_##wc (R, Y, X);				\
		  R##_s = Y##_s;					\
		}							\
	      else if (_FP_FRAC_ZEROP_##wc (R))				\
		{							\
		  R##_e = 0;						\
		  R##_s = (FP_ROUNDMODE == FP_RND_MINF);		\
		  goto sub_done;					\
		}							\
	      goto norm;						\
	    }								\
	sub3:								\
	  if (_FP_FRAC_HIGH_##fs (R) & _FP_IMPLBIT_SH_##fs)		\
	    {								\
	      int _FP_ADD_INTERNAL_diff;				\
	      /* Carry into most significant bit of larger one of X and Y, \
		 canceling it; renormalize.  */				\
	      _FP_FRAC_HIGH_##fs (R) &= _FP_IMPLBIT_SH_##fs - 1;	\
	    norm:							\
	      _FP_FRAC_CLZ_##wc (_FP_ADD_INTERNAL_diff, R);		\
	      _FP_ADD_INTERNAL_diff -= _FP_WFRACXBITS_##fs;		\
	      _FP_FRAC_SLL_##wc (R, _FP_ADD_INTERNAL_diff);		\
	      if (R##_e <= _FP_ADD_INTERNAL_diff)			\
		{							\
		  /* R is denormalized.  */				\
		  _FP_ADD_INTERNAL_diff					\
		    = _FP_ADD_INTERNAL_diff - R##_e + 1;		\
		  _FP_FRAC_SRS_##wc (R, _FP_ADD_INTERNAL_diff,		\
				     _FP_WFRACBITS_##fs);		\
		  R##_e = 0;						\
		}							\
	      else							\
		{							\
		  R##_e -= _FP_ADD_INTERNAL_diff;			\
		  _FP_FRAC_HIGH_##fs (R) &= ~(_FP_W_TYPE) _FP_IMPLBIT_SH_##fs; \
		}							\
	    }								\
	sub_done: ;							\
	}								\
    }									\
  while (0)

#define _FP_ADD(fs, wc, R, X, Y) _FP_ADD_INTERNAL (fs, wc, R, X, Y, '+')
#define _FP_SUB(fs, wc, R, X, Y)					\
  do									\
    {									\
      if (!(Y##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (Y)))	\
	Y##_s ^= 1;							\
      _FP_ADD_INTERNAL (fs, wc, R, X, Y, '-');				\
    }									\
  while (0)


/* Main negation routine.  The input value is raw.  */

#define _FP_NEG(fs, wc, R, X)			\
  do						\
    {						\
      _FP_FRAC_COPY_##wc (R, X);		\
      R##_e = X##_e;				\
      R##_s = 1 ^ X##_s;			\
    }						\
  while (0)


/* Main multiplication routine.  The input values should be cooked.  */

#define _FP_MUL(fs, wc, R, X, Y)				\
  do								\
    {								\
      R##_s = X##_s ^ Y##_s;					\
      R##_e = X##_e + Y##_e + 1;				\
      switch (_FP_CLS_COMBINE (X##_c, Y##_c))			\
	{							\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NORMAL):	\
	  R##_c = FP_CLS_NORMAL;				\
								\
	  _FP_MUL_MEAT_##fs (R, X, Y);				\
								\
	  if (_FP_FRAC_OVERP_##wc (fs, R))			\
	    _FP_FRAC_SRS_##wc (R, 1, _FP_WFRACBITS_##fs);	\
	  else							\
	    R##_e--;						\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NAN):		\
	  _FP_CHOOSENAN (fs, wc, R, X, Y, '*');			\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NORMAL):	\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_ZERO):		\
	  R##_s = X##_s;					\
	  /* FALLTHRU */					\
								\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NORMAL):	\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NORMAL):	\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_ZERO):	\
	  _FP_FRAC_COPY_##wc (R, X);				\
	  R##_c = X##_c;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NAN):	\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NAN):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NAN):		\
	  R##_s = Y##_s;					\
	  /* FALLTHRU */					\
								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_INF):	\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_ZERO):	\
	  _FP_FRAC_COPY_##wc (R, Y);				\
	  R##_c = Y##_c;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_ZERO):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_INF):		\
	  R##_s = _FP_NANSIGN_##fs;				\
	  R##_c = FP_CLS_NAN;					\
	  _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);		\
	  FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_IMZ);	\
	  break;						\
								\
	default:						\
	  _FP_UNREACHABLE;					\
	}							\
    }								\
  while (0)


/* Fused multiply-add.  The input values should be cooked.  */

#define _FP_FMA(fs, wc, dwc, R, X, Y, Z)				\
  do									\
    {									\
      __label__ done_fma;						\
      FP_DECL_##fs (_FP_FMA_T);						\
      _FP_FMA_T##_s = X##_s ^ Y##_s;					\
      _FP_FMA_T##_e = X##_e + Y##_e + 1;				\
      switch (_FP_CLS_COMBINE (X##_c, Y##_c))				\
	{								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NORMAL):		\
	  switch (Z##_c)						\
	    {								\
	    case FP_CLS_INF:						\
	    case FP_CLS_NAN:						\
	      R##_s = Z##_s;						\
	      _FP_FRAC_COPY_##wc (R, Z);				\
	      R##_c = Z##_c;						\
	      break;							\
									\
	    case FP_CLS_ZERO:						\
	      R##_c = FP_CLS_NORMAL;					\
	      R##_s = _FP_FMA_T##_s;					\
	      R##_e = _FP_FMA_T##_e;					\
									\
	      _FP_MUL_MEAT_##fs (R, X, Y);				\
									\
	      if (_FP_FRAC_OVERP_##wc (fs, R))				\
		_FP_FRAC_SRS_##wc (R, 1, _FP_WFRACBITS_##fs);		\
	      else							\
		R##_e--;						\
	      break;							\
									\
	    case FP_CLS_NORMAL:;					\
	      _FP_FRAC_DECL_##dwc (_FP_FMA_TD);				\
	      _FP_FRAC_DECL_##dwc (_FP_FMA_ZD);				\
	      _FP_FRAC_DECL_##dwc (_FP_FMA_RD);				\
	      _FP_MUL_MEAT_DW_##fs (_FP_FMA_TD, X, Y);			\
	      R##_e = _FP_FMA_T##_e;					\
	      int _FP_FMA_tsh						\
		= _FP_FRAC_HIGHBIT_DW_##dwc (fs, _FP_FMA_TD) == 0;	\
	      _FP_FMA_T##_e -= _FP_FMA_tsh;				\
	      int _FP_FMA_ediff = _FP_FMA_T##_e - Z##_e;		\
	      if (_FP_FMA_ediff >= 0)					\
		{							\
		  int _FP_FMA_shift					\
		    = _FP_WFRACBITS_##fs - _FP_FMA_tsh - _FP_FMA_ediff;	\
		  if (_FP_FMA_shift <= -_FP_WFRACBITS_##fs)		\
		    _FP_FRAC_SET_##dwc (_FP_FMA_ZD, _FP_MINFRAC_##dwc);	\
		  else							\
		    {							\
		      _FP_FRAC_COPY_##dwc##_##wc (_FP_FMA_ZD, Z);	\
		      if (_FP_FMA_shift < 0)				\
			_FP_FRAC_SRS_##dwc (_FP_FMA_ZD, -_FP_FMA_shift,	\
					    _FP_WFRACBITS_DW_##fs);	\
		      else if (_FP_FMA_shift > 0)			\
			_FP_FRAC_SLL_##dwc (_FP_FMA_ZD, _FP_FMA_shift);	\
		    }							\
		  R##_s = _FP_FMA_T##_s;				\
		  if (_FP_FMA_T##_s == Z##_s)				\
		    _FP_FRAC_ADD_##dwc (_FP_FMA_RD, _FP_FMA_TD,		\
					_FP_FMA_ZD);			\
		  else							\
		    {							\
		      _FP_FRAC_SUB_##dwc (_FP_FMA_RD, _FP_FMA_TD,	\
					  _FP_FMA_ZD);			\
		      if (_FP_FRAC_NEGP_##dwc (_FP_FMA_RD))		\
			{						\
			  R##_s = Z##_s;				\
			  _FP_FRAC_SUB_##dwc (_FP_FMA_RD, _FP_FMA_ZD,	\
					      _FP_FMA_TD);		\
			}						\
		    }							\
		}							\
	      else							\
		{							\
		  R##_e = Z##_e;					\
		  R##_s = Z##_s;					\
		  _FP_FRAC_COPY_##dwc##_##wc (_FP_FMA_ZD, Z);		\
		  _FP_FRAC_SLL_##dwc (_FP_FMA_ZD, _FP_WFRACBITS_##fs);	\
		  int _FP_FMA_shift = -_FP_FMA_ediff - _FP_FMA_tsh;	\
		  if (_FP_FMA_shift >= _FP_WFRACBITS_DW_##fs)		\
		    _FP_FRAC_SET_##dwc (_FP_FMA_TD, _FP_MINFRAC_##dwc);	\
		  else if (_FP_FMA_shift > 0)				\
		    _FP_FRAC_SRS_##dwc (_FP_FMA_TD, _FP_FMA_shift,	\
					_FP_WFRACBITS_DW_##fs);		\
		  if (Z##_s == _FP_FMA_T##_s)				\
		    _FP_FRAC_ADD_##dwc (_FP_FMA_RD, _FP_FMA_ZD,		\
					_FP_FMA_TD);			\
		  else							\
		    _FP_FRAC_SUB_##dwc (_FP_FMA_RD, _FP_FMA_ZD,		\
					_FP_FMA_TD);			\
		}							\
	      if (_FP_FRAC_ZEROP_##dwc (_FP_FMA_RD))			\
		{							\
		  if (_FP_FMA_T##_s == Z##_s)				\
		    R##_s = Z##_s;					\
		  else							\
		    R##_s = (FP_ROUNDMODE == FP_RND_MINF);		\
		  _FP_FRAC_SET_##wc (R, _FP_ZEROFRAC_##wc);		\
		  R##_c = FP_CLS_ZERO;					\
		}							\
	      else							\
		{							\
		  int _FP_FMA_rlz;					\
		  _FP_FRAC_CLZ_##dwc (_FP_FMA_rlz, _FP_FMA_RD);		\
		  _FP_FMA_rlz -= _FP_WFRACXBITS_DW_##fs;		\
		  R##_e -= _FP_FMA_rlz;					\
		  int _FP_FMA_shift = _FP_WFRACBITS_##fs - _FP_FMA_rlz;	\
		  if (_FP_FMA_shift > 0)				\
		    _FP_FRAC_SRS_##dwc (_FP_FMA_RD, _FP_FMA_shift,	\
					_FP_WFRACBITS_DW_##fs);		\
		  else if (_FP_FMA_shift < 0)				\
		    _FP_FRAC_SLL_##dwc (_FP_FMA_RD, -_FP_FMA_shift);	\
		  _FP_FRAC_COPY_##wc##_##dwc (R, _FP_FMA_RD);		\
		  R##_c = FP_CLS_NORMAL;				\
		}							\
	      break;							\
	    }								\
	  goto done_fma;						\
									\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NAN):			\
	  _FP_CHOOSENAN (fs, wc, _FP_FMA_T, X, Y, '*');			\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_INF):			\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_ZERO):			\
	  _FP_FMA_T##_s = X##_s;					\
	  /* FALLTHRU */						\
									\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_INF):			\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_ZERO):		\
	  _FP_FRAC_COPY_##wc (_FP_FMA_T, X);				\
	  _FP_FMA_T##_c = X##_c;					\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NAN):		\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NAN):			\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NAN):			\
	  _FP_FMA_T##_s = Y##_s;					\
	  /* FALLTHRU */						\
									\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_ZERO):		\
	  _FP_FRAC_COPY_##wc (_FP_FMA_T, Y);				\
	  _FP_FMA_T##_c = Y##_c;					\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_ZERO):			\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_INF):			\
	  _FP_FMA_T##_s = _FP_NANSIGN_##fs;				\
	  _FP_FMA_T##_c = FP_CLS_NAN;					\
	  _FP_FRAC_SET_##wc (_FP_FMA_T, _FP_NANFRAC_##fs);		\
	  FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_IMZ_FMA);	\
	  break;							\
									\
	default:							\
	  _FP_UNREACHABLE;						\
	}								\
									\
      /* T = X * Y is zero, infinity or NaN.  */			\
      switch (_FP_CLS_COMBINE (_FP_FMA_T##_c, Z##_c))			\
	{								\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NAN):			\
	  _FP_CHOOSENAN (fs, wc, R, _FP_FMA_T, Z, '+');			\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_INF):			\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_ZERO):			\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_ZERO):			\
	  R##_s = _FP_FMA_T##_s;					\
	  _FP_FRAC_COPY_##wc (R, _FP_FMA_T);				\
	  R##_c = _FP_FMA_T##_c;					\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NAN):			\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NAN):			\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NORMAL):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_INF):			\
	  R##_s = Z##_s;						\
	  _FP_FRAC_COPY_##wc (R, Z);					\
	  R##_c = Z##_c;						\
	  R##_e = Z##_e;						\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_INF):			\
	  if (_FP_FMA_T##_s == Z##_s)					\
	    {								\
	      R##_s = Z##_s;						\
	      _FP_FRAC_COPY_##wc (R, Z);				\
	      R##_c = Z##_c;						\
	    }								\
	  else								\
	    {								\
	      R##_s = _FP_NANSIGN_##fs;					\
	      R##_c = FP_CLS_NAN;					\
	      _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);			\
	      FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_ISI);	\
	    }								\
	  break;							\
									\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_ZERO):		\
	  if (_FP_FMA_T##_s == Z##_s)					\
	    R##_s = Z##_s;						\
	  else								\
	    R##_s = (FP_ROUNDMODE == FP_RND_MINF);			\
	  _FP_FRAC_COPY_##wc (R, Z);					\
	  R##_c = Z##_c;						\
	  break;							\
									\
	default:							\
	  _FP_UNREACHABLE;						\
	}								\
    done_fma: ;								\
    }									\
  while (0)


/* Main division routine.  The input values should be cooked.  */

#define _FP_DIV(fs, wc, R, X, Y)				\
  do								\
    {								\
      R##_s = X##_s ^ Y##_s;					\
      R##_e = X##_e - Y##_e;					\
      switch (_FP_CLS_COMBINE (X##_c, Y##_c))			\
	{							\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NORMAL):	\
	  R##_c = FP_CLS_NORMAL;				\
								\
	  _FP_DIV_MEAT_##fs (R, X, Y);				\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NAN):		\
	  _FP_CHOOSENAN (fs, wc, R, X, Y, '/');			\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_NORMAL):	\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_NAN, FP_CLS_ZERO):		\
	  R##_s = X##_s;					\
	  _FP_FRAC_COPY_##wc (R, X);				\
	  R##_c = X##_c;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_NAN):	\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NAN):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NAN):		\
	  R##_s = Y##_s;					\
	  _FP_FRAC_COPY_##wc (R, Y);				\
	  R##_c = Y##_c;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_INF):	\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_NORMAL):	\
	  R##_c = FP_CLS_ZERO;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_NORMAL, FP_CLS_ZERO):	\
	  FP_SET_EXCEPTION (FP_EX_DIVZERO);			\
	  /* FALLTHRU */					\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_ZERO):		\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_NORMAL):	\
	  R##_c = FP_CLS_INF;					\
	  break;						\
								\
	case _FP_CLS_COMBINE (FP_CLS_INF, FP_CLS_INF):		\
	case _FP_CLS_COMBINE (FP_CLS_ZERO, FP_CLS_ZERO):	\
	  R##_s = _FP_NANSIGN_##fs;				\
	  R##_c = FP_CLS_NAN;					\
	  _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);		\
	  FP_SET_EXCEPTION (FP_EX_INVALID			\
			    | (X##_c == FP_CLS_INF		\
			       ? FP_EX_INVALID_IDI		\
			       : FP_EX_INVALID_ZDZ));		\
	  break;						\
								\
	default:						\
	  _FP_UNREACHABLE;					\
	}							\
    }								\
  while (0)


/* Helper for comparisons.  EX is 0 not to raise exceptions, 1 to
   raise exceptions for signaling NaN operands, 2 to raise exceptions
   for all NaN operands.  Conditionals are organized to allow the
   compiler to optimize away code based on the value of EX.  */

#define _FP_CMP_CHECK_NAN(fs, wc, X, Y, ex)				\
  do									\
    {									\
      /* The arguments are unordered, which may or may not result in	\
	 an exception.  */						\
      if (ex)								\
	{								\
	  /* At least some cases of unordered arguments result in	\
	     exceptions; check whether this is one.  */			\
	  if (FP_EX_INVALID_SNAN || FP_EX_INVALID_VC)			\
	    {								\
	      /* Check separately for each case of "invalid"		\
		 exceptions.  */					\
	      if ((ex) == 2)						\
		FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_VC);	\
	      if (_FP_ISSIGNAN (fs, wc, X)				\
		  || _FP_ISSIGNAN (fs, wc, Y))				\
		FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_SNAN);	\
	    }								\
	  /* Otherwise, we only need to check whether to raise an	\
	     exception, not which case or cases it is.  */		\
	  else if ((ex) == 2						\
		   || _FP_ISSIGNAN (fs, wc, X)				\
		   || _FP_ISSIGNAN (fs, wc, Y))				\
	    FP_SET_EXCEPTION (FP_EX_INVALID);				\
	}								\
    }									\
  while (0)

/* Helper for comparisons.  If denormal operands would raise an
   exception, check for them, and flush to zero as appropriate
   (otherwise, we need only check and flush to zero if it might affect
   the result, which is done later with _FP_CMP_CHECK_FLUSH_ZERO).  */
#define _FP_CMP_CHECK_DENORM(fs, wc, X, Y)				\
  do									\
    {									\
      if (FP_EX_DENORM != 0)						\
	{								\
	  /* We must ensure the correct exceptions are raised for	\
	     denormal operands, even though this may not affect the	\
	     result of the comparison.  */				\
	  if (FP_DENORM_ZERO)						\
	    {								\
	      _FP_CHECK_FLUSH_ZERO (fs, wc, X);				\
	      _FP_CHECK_FLUSH_ZERO (fs, wc, Y);				\
	    }								\
	  else								\
	    {								\
	      if ((X##_e == 0 && !_FP_FRAC_ZEROP_##wc (X))		\
		  || (Y##_e == 0 && !_FP_FRAC_ZEROP_##wc (Y)))		\
		FP_SET_EXCEPTION (FP_EX_DENORM);			\
	    }								\
	}								\
    }									\
  while (0)

/* Helper for comparisons.  Check for flushing denormals for zero if
   we didn't need to check earlier for any denormal operands.  */
#define _FP_CMP_CHECK_FLUSH_ZERO(fs, wc, X, Y)	\
  do						\
    {						\
      if (FP_EX_DENORM == 0)			\
	{					\
	  _FP_CHECK_FLUSH_ZERO (fs, wc, X);	\
	  _FP_CHECK_FLUSH_ZERO (fs, wc, Y);	\
	}					\
    }						\
  while (0)

/* Main differential comparison routine.  The inputs should be raw not
   cooked.  The return is -1, 0, 1 for normal values, UN
   otherwise.  */

#define _FP_CMP(fs, wc, ret, X, Y, un, ex)				\
  do									\
    {									\
      _FP_CMP_CHECK_DENORM (fs, wc, X, Y);				\
      /* NANs are unordered.  */					\
      if ((X##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (X))	\
	  || (Y##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (Y)))	\
	{								\
	  (ret) = (un);							\
	  _FP_CMP_CHECK_NAN (fs, wc, X, Y, (ex));			\
	}								\
      else								\
	{								\
	  int _FP_CMP_is_zero_x;					\
	  int _FP_CMP_is_zero_y;					\
									\
	  _FP_CMP_CHECK_FLUSH_ZERO (fs, wc, X, Y);			\
									\
	  _FP_CMP_is_zero_x						\
	    = (!X##_e && _FP_FRAC_ZEROP_##wc (X)) ? 1 : 0;		\
	  _FP_CMP_is_zero_y						\
	    = (!Y##_e && _FP_FRAC_ZEROP_##wc (Y)) ? 1 : 0;		\
									\
	  if (_FP_CMP_is_zero_x && _FP_CMP_is_zero_y)			\
	    (ret) = 0;							\
	  else if (_FP_CMP_is_zero_x)					\
	    (ret) = Y##_s ? 1 : -1;					\
	  else if (_FP_CMP_is_zero_y)					\
	    (ret) = X##_s ? -1 : 1;					\
	  else if (X##_s != Y##_s)					\
	    (ret) = X##_s ? -1 : 1;					\
	  else if (X##_e > Y##_e)					\
	    (ret) = X##_s ? -1 : 1;					\
	  else if (X##_e < Y##_e)					\
	    (ret) = X##_s ? 1 : -1;					\
	  else if (_FP_FRAC_GT_##wc (X, Y))				\
	    (ret) = X##_s ? -1 : 1;					\
	  else if (_FP_FRAC_GT_##wc (Y, X))				\
	    (ret) = X##_s ? 1 : -1;					\
	  else								\
	    (ret) = 0;							\
	}								\
    }									\
  while (0)


/* Simplification for strict equality.  */

#define _FP_CMP_EQ(fs, wc, ret, X, Y, ex)				\
  do									\
    {									\
      _FP_CMP_CHECK_DENORM (fs, wc, X, Y);				\
      /* NANs are unordered.  */					\
      if ((X##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (X))	\
	  || (Y##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (Y)))	\
	{								\
	  (ret) = 1;							\
	  _FP_CMP_CHECK_NAN (fs, wc, X, Y, (ex));			\
	}								\
      else								\
	{								\
	  _FP_CMP_CHECK_FLUSH_ZERO (fs, wc, X, Y);			\
									\
	  (ret) = !(X##_e == Y##_e					\
		    && _FP_FRAC_EQ_##wc (X, Y)				\
		    && (X##_s == Y##_s					\
			|| (!X##_e && _FP_FRAC_ZEROP_##wc (X))));	\
	}								\
    }									\
  while (0)

/* Version to test unordered.  */

#define _FP_CMP_UNORD(fs, wc, ret, X, Y, ex)				\
  do									\
    {									\
      _FP_CMP_CHECK_DENORM (fs, wc, X, Y);				\
      (ret) = ((X##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (X))	\
	       || (Y##_e == _FP_EXPMAX_##fs && !_FP_FRAC_ZEROP_##wc (Y))); \
      if (ret)								\
	_FP_CMP_CHECK_NAN (fs, wc, X, Y, (ex));				\
    }									\
  while (0)

/* Main square root routine.  The input value should be cooked.  */

#define _FP_SQRT(fs, wc, R, X)						\
  do									\
    {									\
      _FP_FRAC_DECL_##wc (_FP_SQRT_T);					\
      _FP_FRAC_DECL_##wc (_FP_SQRT_S);					\
      _FP_W_TYPE _FP_SQRT_q;						\
      switch (X##_c)							\
	{								\
	case FP_CLS_NAN:						\
	  _FP_FRAC_COPY_##wc (R, X);					\
	  R##_s = X##_s;						\
	  R##_c = FP_CLS_NAN;						\
	  break;							\
	case FP_CLS_INF:						\
	  if (X##_s)							\
	    {								\
	      R##_s = _FP_NANSIGN_##fs;					\
	      R##_c = FP_CLS_NAN; /* NAN */				\
	      _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);			\
	      FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_SQRT);	\
	    }								\
	  else								\
	    {								\
	      R##_s = 0;						\
	      R##_c = FP_CLS_INF; /* sqrt(+inf) = +inf */		\
	    }								\
	  break;							\
	case FP_CLS_ZERO:						\
	  R##_s = X##_s;						\
	  R##_c = FP_CLS_ZERO; /* sqrt(+-0) = +-0 */			\
	  break;							\
	case FP_CLS_NORMAL:						\
	  R##_s = 0;							\
	  if (X##_s)							\
	    {								\
	      R##_c = FP_CLS_NAN; /* NAN */				\
	      R##_s = _FP_NANSIGN_##fs;					\
	      _FP_FRAC_SET_##wc (R, _FP_NANFRAC_##fs);			\
	      FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_SQRT);	\
	      break;							\
	    }								\
	  R##_c = FP_CLS_NORMAL;					\
	  if (X##_e & 1)						\
	    _FP_FRAC_SLL_##wc (X, 1);					\
	  R##_e = X##_e >> 1;						\
	  _FP_FRAC_SET_##wc (_FP_SQRT_S, _FP_ZEROFRAC_##wc);		\
	  _FP_FRAC_SET_##wc (R, _FP_ZEROFRAC_##wc);			\
	  _FP_SQRT_q = _FP_OVERFLOW_##fs >> 1;				\
	  _FP_SQRT_MEAT_##wc (R, _FP_SQRT_S, _FP_SQRT_T, X,		\
			      _FP_SQRT_q);				\
	}								\
    }									\
  while (0)

/* Convert from FP to integer.  Input is raw.  */

/* RSIGNED can have following values:
   0:  the number is required to be 0..(2^rsize)-1, if not, NV is set plus
       the result is either 0 or (2^rsize)-1 depending on the sign in such
       case.
   1:  the number is required to be -(2^(rsize-1))..(2^(rsize-1))-1, if not,
       NV is set plus the result is either -(2^(rsize-1)) or (2^(rsize-1))-1
       depending on the sign in such case.
   2:  the number is required to be -(2^(rsize-1))..(2^(rsize-1))-1, if not,
       NV is set plus the result is reduced modulo 2^rsize.
   -1: the number is required to be -(2^(rsize-1))..(2^rsize)-1, if not, NV is
       set plus the result is either -(2^(rsize-1)) or (2^(rsize-1))-1
       depending on the sign in such case.  */
#define _FP_TO_INT(fs, wc, r, X, rsize, rsigned)			\
  do									\
    {									\
      if (X##_e < _FP_EXPBIAS_##fs)					\
	{								\
	  (r) = 0;							\
	  if (X##_e == 0)						\
	    {								\
	      if (!_FP_FRAC_ZEROP_##wc (X))				\
		{							\
		  if (!FP_DENORM_ZERO)					\
		    FP_SET_EXCEPTION (FP_EX_INEXACT);			\
		  FP_SET_EXCEPTION (FP_EX_DENORM);			\
		}							\
	    }								\
	  else								\
	    FP_SET_EXCEPTION (FP_EX_INEXACT);				\
	}								\
      else if ((rsigned) == 2						\
	       && (X##_e						\
		   >= ((_FP_EXPMAX_##fs					\
			< _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs + (rsize) - 1) \
		       ? _FP_EXPMAX_##fs				\
		       : _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs + (rsize) - 1))) \
	{								\
	  /* Overflow resulting in 0.  */				\
	  (r) = 0;							\
	  FP_SET_EXCEPTION (FP_EX_INVALID				\
			    | FP_EX_INVALID_CVI				\
			    | ((FP_EX_INVALID_SNAN			\
				&& _FP_ISSIGNAN (fs, wc, X))		\
			       ? FP_EX_INVALID_SNAN			\
			       : 0));					\
	}								\
      else if ((rsigned) != 2						\
	       && (X##_e >= (_FP_EXPMAX_##fs < _FP_EXPBIAS_##fs + (rsize) \
			     ? _FP_EXPMAX_##fs				\
			     : (_FP_EXPBIAS_##fs + (rsize)		\
				- ((rsigned) > 0 || X##_s)))		\
		   || (!(rsigned) && X##_s)))				\
	{								\
	  /* Overflow or converting to the most negative integer.  */	\
	  if (rsigned)							\
	    {								\
	      (r) = 1;							\
	      (r) <<= (rsize) - 1;					\
	      (r) -= 1 - X##_s;						\
	    }								\
	  else								\
	    {								\
	      (r) = 0;							\
	      if (!X##_s)						\
		(r) = ~(r);						\
	    }								\
									\
	  if (_FP_EXPBIAS_##fs + (rsize) - 1 < _FP_EXPMAX_##fs		\
	      && (rsigned)						\
	      && X##_s							\
	      && X##_e == _FP_EXPBIAS_##fs + (rsize) - 1)		\
	    {								\
	      /* Possibly converting to most negative integer; check the \
		 mantissa.  */						\
	      int _FP_TO_INT_inexact = 0;				\
	      (void) ((_FP_FRACBITS_##fs > (rsize))			\
		      ? ({						\
			  _FP_FRAC_SRST_##wc (X, _FP_TO_INT_inexact,	\
					      _FP_FRACBITS_##fs - (rsize), \
					      _FP_FRACBITS_##fs);	\
			  0;						\
			})						\
		      : 0);						\
	      if (!_FP_FRAC_ZEROP_##wc (X))				\
		FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_CVI);	\
	      else if (_FP_TO_INT_inexact)				\
		FP_SET_EXCEPTION (FP_EX_INEXACT);			\
	    }								\
	  else								\
	    FP_SET_EXCEPTION (FP_EX_INVALID				\
			      | FP_EX_INVALID_CVI			\
			      | ((FP_EX_INVALID_SNAN			\
				  && _FP_ISSIGNAN (fs, wc, X))		\
				 ? FP_EX_INVALID_SNAN			\
				 : 0));					\
	}								\
      else								\
	{								\
	  int _FP_TO_INT_inexact = 0;					\
	  _FP_FRAC_HIGH_RAW_##fs (X) |= _FP_IMPLBIT_##fs;		\
	  if (X##_e >= _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs - 1)	\
	    {								\
	      _FP_FRAC_ASSEMBLE_##wc ((r), X, (rsize));			\
	      (r) <<= X##_e - _FP_EXPBIAS_##fs - _FP_FRACBITS_##fs + 1; \
	    }								\
	  else								\
	    {								\
	      _FP_FRAC_SRST_##wc (X, _FP_TO_INT_inexact,		\
				  (_FP_FRACBITS_##fs + _FP_EXPBIAS_##fs - 1 \
				   - X##_e),				\
				  _FP_FRACBITS_##fs);			\
	      _FP_FRAC_ASSEMBLE_##wc ((r), X, (rsize));			\
	    }								\
	  if ((rsigned) && X##_s)					\
	    (r) = -(r);							\
	  if ((rsigned) == 2 && X##_e >= _FP_EXPBIAS_##fs + (rsize) - 1) \
	    {								\
	      /* Overflow or converting to the most negative integer.  */ \
	      if (X##_e > _FP_EXPBIAS_##fs + (rsize) - 1		\
		  || !X##_s						\
		  || (r) != (((typeof (r)) 1) << ((rsize) - 1)))	\
		{							\
		  _FP_TO_INT_inexact = 0;				\
		  FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_CVI);	\
		}							\
	    }								\
	  if (_FP_TO_INT_inexact)					\
	    FP_SET_EXCEPTION (FP_EX_INEXACT);				\
	}								\
    }									\
  while (0)

/* Convert from floating point to integer, rounding according to the
   current rounding direction.  Input is raw.  RSIGNED is as for
   _FP_TO_INT.  */
#define _FP_TO_INT_ROUND(fs, wc, r, X, rsize, rsigned)			\
  do									\
    {									\
      __label__ _FP_TO_INT_ROUND_done;					\
      if (X##_e < _FP_EXPBIAS_##fs)					\
	{								\
	  int _FP_TO_INT_ROUND_rounds_away = 0;				\
	  if (X##_e == 0)						\
	    {								\
	      if (_FP_FRAC_ZEROP_##wc (X))				\
		{							\
		  (r) = 0;						\
		  goto _FP_TO_INT_ROUND_done;				\
		}							\
	      else							\
		{							\
		  FP_SET_EXCEPTION (FP_EX_DENORM);			\
		  if (FP_DENORM_ZERO)					\
		    {							\
		      (r) = 0;						\
		      goto _FP_TO_INT_ROUND_done;			\
		    }							\
		}							\
	    }								\
	  /* The result is 0, 1 or -1 depending on the rounding mode;	\
	     -1 may cause overflow in the unsigned case.  */		\
	  switch (FP_ROUNDMODE)						\
	    {								\
	    case FP_RND_NEAREST:					\
	      _FP_TO_INT_ROUND_rounds_away				\
		= (X##_e == _FP_EXPBIAS_##fs - 1			\
		   && !_FP_FRAC_ZEROP_##wc (X));			\
	      break;							\
	    case FP_RND_ZERO:						\
	      /* _FP_TO_INT_ROUND_rounds_away is already 0.  */		\
	      break;							\
	    case FP_RND_PINF:						\
	      _FP_TO_INT_ROUND_rounds_away = !X##_s;			\
	      break;							\
	    case FP_RND_MINF:						\
	      _FP_TO_INT_ROUND_rounds_away = X##_s;			\
	      break;							\
	    }								\
	  if ((rsigned) == 0 && _FP_TO_INT_ROUND_rounds_away && X##_s)	\
	    {								\
	      /* Result of -1 for an unsigned conversion.  */		\
	      (r) = 0;							\
	      FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_CVI);	\
	    }								\
	  else if ((rsize) == 1 && (rsigned) > 0			\
		   && _FP_TO_INT_ROUND_rounds_away && !X##_s)		\
	    {								\
	      /* Converting to a 1-bit signed bit-field, which cannot	\
		 represent +1.  */					\
	      (r) = ((rsigned) == 2 ? -1 : 0);				\
	      FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_CVI);	\
	    }								\
	  else								\
	    {								\
	      (r) = (_FP_TO_INT_ROUND_rounds_away			\
		     ? (X##_s ? -1 : 1)					\
		     : 0);						\
	      FP_SET_EXCEPTION (FP_EX_INEXACT);				\
	    }								\
	}								\
      else if ((rsigned) == 2						\
	       && (X##_e						\
		   >= ((_FP_EXPMAX_##fs					\
			< _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs + (rsize) - 1) \
		       ? _FP_EXPMAX_##fs				\
		       : _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs + (rsize) - 1))) \
	{								\
	  /* Overflow resulting in 0.  */				\
	  (r) = 0;							\
	  FP_SET_EXCEPTION (FP_EX_INVALID				\
			    | FP_EX_INVALID_CVI				\
			    | ((FP_EX_INVALID_SNAN			\
				&& _FP_ISSIGNAN (fs, wc, X))		\
			       ? FP_EX_INVALID_SNAN			\
			       : 0));					\
	}								\
      else if ((rsigned) != 2						\
	       && (X##_e >= (_FP_EXPMAX_##fs < _FP_EXPBIAS_##fs + (rsize) \
			     ? _FP_EXPMAX_##fs				\
			     : (_FP_EXPBIAS_##fs + (rsize)		\
				- ((rsigned) > 0 && !X##_s)))		\
		   || ((rsigned) == 0 && X##_s)))			\
	{								\
	  /* Definite overflow (does not require rounding to tell).  */	\
	  if ((rsigned) != 0)						\
	    {								\
	      (r) = 1;							\
	      (r) <<= (rsize) - 1;					\
	      (r) -= 1 - X##_s;						\
	    }								\
	  else								\
	    {								\
	      (r) = 0;							\
	      if (!X##_s)						\
		(r) = ~(r);						\
	    }								\
									\
	  FP_SET_EXCEPTION (FP_EX_INVALID				\
			    | FP_EX_INVALID_CVI				\
			    | ((FP_EX_INVALID_SNAN			\
				&& _FP_ISSIGNAN (fs, wc, X))		\
			       ? FP_EX_INVALID_SNAN			\
			       : 0));					\
	}								\
      else								\
	{								\
	  /* The value is finite, with magnitude at least 1.  If	\
	     the conversion is unsigned, the value is positive.		\
	     If RSIGNED is not 2, the value does not definitely		\
	     overflow by virtue of its exponent, but may still turn	\
	     out to overflow after rounding; if RSIGNED is 2, the	\
	     exponent may be such that the value definitely overflows,	\
	     but at least one mantissa bit will not be shifted out.  */ \
	  int _FP_TO_INT_ROUND_inexact = 0;				\
	  _FP_FRAC_HIGH_RAW_##fs (X) |= _FP_IMPLBIT_##fs;		\
	  if (X##_e >= _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs - 1)	\
	    {								\
	      /* The value is an integer, no rounding needed.  */	\
	      _FP_FRAC_ASSEMBLE_##wc ((r), X, (rsize));			\
	      (r) <<= X##_e - _FP_EXPBIAS_##fs - _FP_FRACBITS_##fs + 1; \
	    }								\
	  else								\
	    {								\
	      /* May need to shift in order to round (unless there	\
		 are exactly _FP_WORKBITS fractional bits already).  */	\
	      int _FP_TO_INT_ROUND_rshift				\
		= (_FP_FRACBITS_##fs + _FP_EXPBIAS_##fs			\
		   - 1 - _FP_WORKBITS - X##_e);				\
	      if (_FP_TO_INT_ROUND_rshift > 0)				\
		_FP_FRAC_SRS_##wc (X, _FP_TO_INT_ROUND_rshift,		\
				   _FP_WFRACBITS_##fs);			\
	      else if (_FP_TO_INT_ROUND_rshift < 0)			\
		_FP_FRAC_SLL_##wc (X, -_FP_TO_INT_ROUND_rshift);	\
	      /* Round like _FP_ROUND, but setting			\
		 _FP_TO_INT_ROUND_inexact instead of directly setting	\
		 the "inexact" exception, since it may turn out we	\
		 should set "invalid" instead.  */			\
	      if (_FP_FRAC_LOW_##wc (X) & 7)				\
		{							\
		  _FP_TO_INT_ROUND_inexact = 1;				\
		  switch (FP_ROUNDMODE)					\
		    {							\
		    case FP_RND_NEAREST:				\
		      _FP_ROUND_NEAREST (wc, X);			\
		      break;						\
		    case FP_RND_ZERO:					\
		      _FP_ROUND_ZERO (wc, X);				\
		      break;						\
		    case FP_RND_PINF:					\
		      _FP_ROUND_PINF (wc, X);				\
		      break;						\
		    case FP_RND_MINF:					\
		      _FP_ROUND_MINF (wc, X);				\
		      break;						\
		    }							\
		}							\
	      _FP_FRAC_SRL_##wc (X, _FP_WORKBITS);			\
	      _FP_FRAC_ASSEMBLE_##wc ((r), X, (rsize));			\
	    }								\
	  if ((rsigned) != 0 && X##_s)					\
	    (r) = -(r);							\
	  /* An exponent of RSIZE - 1 always needs testing for		\
	     overflow (either directly overflowing, or overflowing	\
	     when rounding up results in 2^RSIZE).  An exponent of	\
	     RSIZE - 2 can overflow for positive values when rounding	\
	     up to 2^(RSIZE-1), but cannot overflow for negative	\
	     values.  Smaller exponents cannot overflow.  */		\
	  if (X##_e >= (_FP_EXPBIAS_##fs + (rsize) - 1			\
			- ((rsigned) > 0 && !X##_s)))			\
	    {								\
	      if (X##_e > _FP_EXPBIAS_##fs + (rsize) - 1		\
		  || (X##_e == _FP_EXPBIAS_##fs + (rsize) - 1		\
		      && (X##_s						\
			  ? (r) != (((typeof (r)) 1) << ((rsize) - 1))	\
			  : ((rsigned) > 0 || (r) == 0)))		\
		  || ((rsigned) > 0					\
		      && !X##_s						\
		      && X##_e == _FP_EXPBIAS_##fs + (rsize) - 2	\
		      && (r) == (((typeof (r)) 1) << ((rsize) - 1))))	\
		{							\
		  if ((rsigned) != 2)					\
		    {							\
		      if ((rsigned) != 0)				\
			{						\
			  (r) = 1;					\
			  (r) <<= (rsize) - 1;				\
			  (r) -= 1 - X##_s;				\
			}						\
		      else						\
			{						\
			  (r) = 0;					\
			  (r) = ~(r);					\
			}						\
		    }							\
		  _FP_TO_INT_ROUND_inexact = 0;				\
		  FP_SET_EXCEPTION (FP_EX_INVALID | FP_EX_INVALID_CVI);	\
		}							\
	    }								\
	  if (_FP_TO_INT_ROUND_inexact)					\
	    FP_SET_EXCEPTION (FP_EX_INEXACT);				\
	}								\
    _FP_TO_INT_ROUND_done: ;						\
    }									\
  while (0)

/* Convert integer to fp.  Output is raw.  RTYPE is unsigned even if
   input is signed.  */
#define _FP_FROM_INT(fs, wc, X, r, rsize, rtype)			\
  do									\
    {									\
      __label__ pack_semiraw;						\
      if (r)								\
	{								\
	  rtype _FP_FROM_INT_ur = (r);					\
									\
	  if ((X##_s = ((r) < 0)))					\
	    _FP_FROM_INT_ur = -_FP_FROM_INT_ur;				\
									\
	  _FP_STATIC_ASSERT ((rsize) <= 2 * _FP_W_TYPE_SIZE,		\
			     "rsize too large");			\
	  (void) (((rsize) <= _FP_W_TYPE_SIZE)				\
		  ? ({							\
		      int _FP_FROM_INT_lz;				\
		      __FP_CLZ (_FP_FROM_INT_lz,			\
				(_FP_W_TYPE) _FP_FROM_INT_ur);		\
		      X##_e = (_FP_EXPBIAS_##fs + _FP_W_TYPE_SIZE - 1	\
			       - _FP_FROM_INT_lz);			\
		    })							\
		  : ({						\
		      int _FP_FROM_INT_lz;				\
		      __FP_CLZ_2 (_FP_FROM_INT_lz,			\
				  (_FP_W_TYPE) (_FP_FROM_INT_ur		\
						>> _FP_W_TYPE_SIZE),	\
				  (_FP_W_TYPE) _FP_FROM_INT_ur);	\
		      X##_e = (_FP_EXPBIAS_##fs + 2 * _FP_W_TYPE_SIZE - 1 \
			       - _FP_FROM_INT_lz);			\
		    }));						\
									\
	  if ((rsize) - 1 + _FP_EXPBIAS_##fs >= _FP_EXPMAX_##fs		\
	      && X##_e >= _FP_EXPMAX_##fs)				\
	    {								\
	      /* Exponent too big; overflow to infinity.  (May also	\
		 happen after rounding below.)  */			\
	      _FP_OVERFLOW_SEMIRAW (fs, wc, X);				\
	      goto pack_semiraw;					\
	    }								\
									\
	  if ((rsize) <= _FP_FRACBITS_##fs				\
	      || X##_e < _FP_EXPBIAS_##fs + _FP_FRACBITS_##fs)		\
	    {								\
	      /* Exactly representable; shift left.  */			\
	      _FP_FRAC_DISASSEMBLE_##wc (X, _FP_FROM_INT_ur, (rsize));	\
	      if (_FP_EXPBIAS_##fs + _FP_FRACBITS_##fs - 1 - X##_e > 0)	\
		_FP_FRAC_SLL_##wc (X, (_FP_EXPBIAS_##fs			\
				       + _FP_FRACBITS_##fs - 1 - X##_e)); \
	    }								\
	  else								\
	    {								\
	      /* More bits in integer than in floating type; need to	\
		 round.  */						\
	      if (_FP_EXPBIAS_##fs + _FP_WFRACBITS_##fs - 1 < X##_e)	\
		_FP_FROM_INT_ur						\
		  = ((_FP_FROM_INT_ur >> (X##_e - _FP_EXPBIAS_##fs	\
					  - _FP_WFRACBITS_##fs + 1))	\
		     | ((_FP_FROM_INT_ur				\
			 << ((rsize) - (X##_e - _FP_EXPBIAS_##fs	\
					- _FP_WFRACBITS_##fs + 1)))	\
			!= 0));						\
	      _FP_FRAC_DISASSEMBLE_##wc (X, _FP_FROM_INT_ur, (rsize));	\
	      if ((_FP_EXPBIAS_##fs + _FP_WFRACBITS_##fs - 1 - X##_e) > 0) \
		_FP_FRAC_SLL_##wc (X, (_FP_EXPBIAS_##fs			\
				       + _FP_WFRACBITS_##fs - 1 - X##_e)); \
	      _FP_FRAC_HIGH_##fs (X) &= ~(_FP_W_TYPE) _FP_IMPLBIT_SH_##fs; \
	    pack_semiraw:						\
	      _FP_PACK_SEMIRAW (fs, wc, X);				\
	    }								\
	}								\
      else								\
	{								\
	  X##_s = 0;							\
	  X##_e = 0;							\
	  _FP_FRAC_SET_##wc (X, _FP_ZEROFRAC_##wc);			\
	}								\
    }									\
  while (0)


/* Extend from a narrower floating-point format to a wider one.  Input
   and output are raw.  If CHECK_NAN, then signaling NaNs are
   converted to quiet with the "invalid" exception raised; otherwise
   signaling NaNs remain signaling with no exception.  */
#define _FP_EXTEND_CNAN(dfs, sfs, dwc, swc, D, S, check_nan)		\
  do									\
    {									\
      _FP_STATIC_ASSERT (_FP_FRACBITS_##dfs >= _FP_FRACBITS_##sfs,	\
			 "destination mantissa narrower than source");	\
      _FP_STATIC_ASSERT ((_FP_EXPMAX_##dfs - _FP_EXPBIAS_##dfs		\
			  >= _FP_EXPMAX_##sfs - _FP_EXPBIAS_##sfs),	\
			 "destination max exponent smaller"		\
			 " than source");				\
      _FP_STATIC_ASSERT (((_FP_EXPBIAS_##dfs				\
			   >= (_FP_EXPBIAS_##sfs			\
			       + _FP_FRACBITS_##sfs - 1))		\
			  || (_FP_EXPBIAS_##dfs == _FP_EXPBIAS_##sfs)), \
			 "source subnormals do not all become normal,"	\
			 " but bias not the same");			\
      D##_s = S##_s;							\
      _FP_FRAC_COPY_##dwc##_##swc (D, S);				\
      if (_FP_EXP_NORMAL (sfs, swc, S))					\
	{								\
	  D##_e = S##_e + _FP_EXPBIAS_##dfs - _FP_EXPBIAS_##sfs;	\
	  _FP_FRAC_SLL_##dwc (D, (_FP_FRACBITS_##dfs - _FP_FRACBITS_##sfs)); \
	}								\
      else								\
	{								\
	  if (S##_e == 0)						\
	    {								\
	      _FP_CHECK_FLUSH_ZERO (sfs, swc, S);			\
	      if (_FP_FRAC_ZEROP_##swc (S))				\
		D##_e = 0;						\
	      else if (_FP_EXPBIAS_##dfs				\
		       < _FP_EXPBIAS_##sfs + _FP_FRACBITS_##sfs - 1)	\
		{							\
		  FP_SET_EXCEPTION (FP_EX_DENORM);			\
		  _FP_FRAC_SLL_##dwc (D, (_FP_FRACBITS_##dfs		\
					  - _FP_FRACBITS_##sfs));	\
		  D##_e = 0;						\
		  if (FP_TRAPPING_EXCEPTIONS & FP_EX_UNDERFLOW)		\
		    FP_SET_EXCEPTION (FP_EX_UNDERFLOW);			\
		}							\
	      else							\
		{							\
		  int FP_EXTEND_lz;					\
		  FP_SET_EXCEPTION (FP_EX_DENORM);			\
		  _FP_FRAC_CLZ_##swc (FP_EXTEND_lz, S);			\
		  _FP_FRAC_SLL_##dwc (D,				\
				      FP_EXTEND_lz + _FP_FRACBITS_##dfs	\
				      - _FP_FRACTBITS_##sfs);		\
		  D##_e = (_FP_EXPBIAS_##dfs - _FP_EXPBIAS_##sfs + 1	\
			   + _FP_FRACXBITS_##sfs - FP_EXTEND_lz);	\
		}							\
	    }								\
	  else								\
	    {								\
	      D##_e = _FP_EXPMAX_##dfs;					\
	      if (!_FP_FRAC_ZEROP_##swc (S))				\
		{							\
		  if (check_nan && _FP_FRAC_SNANP (sfs, S))		\
		    FP_SET_EXCEPTION (FP_EX_INVALID			\
				      | FP_EX_INVALID_SNAN);		\
		  _FP_FRAC_SLL_##dwc (D, (_FP_FRACBITS_##dfs		\
					  - _FP_FRACBITS_##sfs));	\
		  if (check_nan)					\
		    _FP_SETQNAN (dfs, dwc, D);				\
		}							\
	    }								\
	}								\
    }									\
  while (0)

#define FP_EXTEND(dfs, sfs, dwc, swc, D, S)		\
    _FP_EXTEND_CNAN (dfs, sfs, dwc, swc, D, S, 1)

/* Truncate from a wider floating-point format to a narrower one.
   Input and output are semi-raw.  */
#define FP_TRUNC(dfs, sfs, dwc, swc, D, S)				\
  do									\
    {									\
      _FP_STATIC_ASSERT (_FP_FRACBITS_##sfs >= _FP_FRACBITS_##dfs,	\
			 "destination mantissa wider than source");	\
      _FP_STATIC_ASSERT (((_FP_EXPBIAS_##sfs				\
			   >= (_FP_EXPBIAS_##dfs			\
			       + _FP_FRACBITS_##dfs - 1))		\
			  || _FP_EXPBIAS_##sfs == _FP_EXPBIAS_##dfs),	\
			 "source subnormals do not all become same,"	\
			 " but bias not the same");			\
      D##_s = S##_s;							\
      if (_FP_EXP_NORMAL (sfs, swc, S))					\
	{								\
	  D##_e = S##_e + _FP_EXPBIAS_##dfs - _FP_EXPBIAS_##sfs;	\
	  if (D##_e >= _FP_EXPMAX_##dfs)				\
	    _FP_OVERFLOW_SEMIRAW (dfs, dwc, D);				\
	  else								\
	    {								\
	      if (D##_e <= 0)						\
		{							\
		  if (D##_e < 1 - _FP_FRACBITS_##dfs)			\
		    {							\
		      _FP_FRAC_SET_##swc (S, _FP_ZEROFRAC_##swc);	\
		      _FP_FRAC_LOW_##swc (S) |= 1;			\
		    }							\
		  else							\
		    {							\
		      _FP_FRAC_HIGH_##sfs (S) |= _FP_IMPLBIT_SH_##sfs;	\
		      _FP_FRAC_SRS_##swc (S, (_FP_WFRACBITS_##sfs	\
					      - _FP_WFRACBITS_##dfs	\
					      + 1 - D##_e),		\
					  _FP_WFRACBITS_##sfs);		\
		    }							\
		  D##_e = 0;						\
		}							\
	      else							\
		_FP_FRAC_SRS_##swc (S, (_FP_WFRACBITS_##sfs		\
					- _FP_WFRACBITS_##dfs),		\
				    _FP_WFRACBITS_##sfs);		\
	      _FP_FRAC_COPY_##dwc##_##swc (D, S);			\
	    }								\
	}								\
      else								\
	{								\
	  if (S##_e == 0)						\
	    {								\
	      _FP_CHECK_FLUSH_ZERO (sfs, swc, S);			\
	      D##_e = 0;						\
	      if (_FP_FRAC_ZEROP_##swc (S))				\
		_FP_FRAC_SET_##dwc (D, _FP_ZEROFRAC_##dwc);		\
	      else							\
		{							\
		  FP_SET_EXCEPTION (FP_EX_DENORM);			\
		  if (_FP_EXPBIAS_##sfs					\
		      < _FP_EXPBIAS_##dfs + _FP_FRACBITS_##dfs - 1)	\
		    {							\
		      _FP_FRAC_SRS_##swc (S, (_FP_WFRACBITS_##sfs	\
					      - _FP_WFRACBITS_##dfs),	\
					  _FP_WFRACBITS_##sfs);		\
		      _FP_FRAC_COPY_##dwc##_##swc (D, S);		\
		    }							\
		  else							\
		    {							\
		      _FP_FRAC_SET_##dwc (D, _FP_ZEROFRAC_##dwc);	\
		      _FP_FRAC_LOW_##dwc (D) |= 1;			\
		    }							\
		}							\
	    }								\
	  else								\
	    {								\
	      D##_e = _FP_EXPMAX_##dfs;					\
	      if (_FP_FRAC_ZEROP_##swc (S))				\
		_FP_FRAC_SET_##dwc (D, _FP_ZEROFRAC_##dwc);		\
	      else							\
		{							\
		  _FP_CHECK_SIGNAN_SEMIRAW (sfs, swc, S);		\
		  _FP_FRAC_SRL_##swc (S, (_FP_WFRACBITS_##sfs		\
					  - _FP_WFRACBITS_##dfs));	\
		  _FP_FRAC_COPY_##dwc##_##swc (D, S);			\
		  /* Semi-raw NaN must have all workbits cleared.  */	\
		  _FP_FRAC_LOW_##dwc (D)				\
		    &= ~(_FP_W_TYPE) ((1 << _FP_WORKBITS) - 1);		\
		  _FP_SETQNAN_SEMIRAW (dfs, dwc, D);			\
		}							\
	    }								\
	}								\
    }									\
  while (0)

/* Truncate from a wider floating-point format to a narrower one.
   Input and output are cooked.  */
#define FP_TRUNC_COOKED(dfs, sfs, dwc, swc, D, S)			\
  do									\
    {									\
      _FP_STATIC_ASSERT (_FP_FRACBITS_##sfs >= _FP_FRACBITS_##dfs,	\
			 "destination mantissa wider than source");	\
      if (S##_c == FP_CLS_NAN)						\
	_FP_FRAC_SRL_##swc (S, (_FP_WFRACBITS_##sfs			\
				- _FP_WFRACBITS_##dfs));		\
      else								\
	_FP_FRAC_SRS_##swc (S, (_FP_WFRACBITS_##sfs			\
				- _FP_WFRACBITS_##dfs),			\
			    _FP_WFRACBITS_##sfs);			\
      _FP_FRAC_COPY_##dwc##_##swc (D, S);				\
      D##_e = S##_e;							\
      D##_c = S##_c;							\
      D##_s = S##_s;							\
    }									\
  while (0)

/* Helper primitives.  */

/* Count leading zeros in a word.  */

#ifndef __FP_CLZ
/* GCC 3.4 and later provide the builtins for us.  */
# define __FP_CLZ(r, x)							\
  do									\
    {									\
      _FP_STATIC_ASSERT ((sizeof (_FP_W_TYPE) == sizeof (unsigned int)	\
			  || (sizeof (_FP_W_TYPE)			\
			      == sizeof (unsigned long))		\
			  || (sizeof (_FP_W_TYPE)			\
			      == sizeof (unsigned long long))),		\
			 "_FP_W_TYPE size unsupported for clz");	\
      if (sizeof (_FP_W_TYPE) == sizeof (unsigned int))			\
	(r) = __builtin_clz (x);					\
      else if (sizeof (_FP_W_TYPE) == sizeof (unsigned long))		\
	(r) = __builtin_clzl (x);					\
      else /* sizeof (_FP_W_TYPE) == sizeof (unsigned long long).  */	\
	(r) = __builtin_clzll (x);					\
    }									\
  while (0)
#endif /* ndef __FP_CLZ */

#define _FP_DIV_HELP_imm(q, r, n, d)		\
  do						\
    {						\
      (q) = (n) / (d), (r) = (n) % (d);		\
    }						\
  while (0)


/* A restoring bit-by-bit division primitive.  */

#define _FP_DIV_MEAT_N_loop(fs, wc, R, X, Y)				\
  do									\
    {									\
      int _FP_DIV_MEAT_N_loop_count = _FP_WFRACBITS_##fs;		\
      _FP_FRAC_DECL_##wc (_FP_DIV_MEAT_N_loop_u);			\
      _FP_FRAC_DECL_##wc (_FP_DIV_MEAT_N_loop_v);			\
      _FP_FRAC_COPY_##wc (_FP_DIV_MEAT_N_loop_u, X);			\
      _FP_FRAC_COPY_##wc (_FP_DIV_MEAT_N_loop_v, Y);			\
      _FP_FRAC_SET_##wc (R, _FP_ZEROFRAC_##wc);				\
      /* Normalize _FP_DIV_MEAT_N_LOOP_U and _FP_DIV_MEAT_N_LOOP_V.  */	\
      _FP_FRAC_SLL_##wc (_FP_DIV_MEAT_N_loop_u, _FP_WFRACXBITS_##fs);	\
      _FP_FRAC_SLL_##wc (_FP_DIV_MEAT_N_loop_v, _FP_WFRACXBITS_##fs);	\
      /* First round.  Since the operands are normalized, either the	\
	 first or second bit will be set in the fraction.  Produce a	\
	 normalized result by checking which and adjusting the loop	\
	 count and exponent accordingly.  */				\
      if (_FP_FRAC_GE_1 (_FP_DIV_MEAT_N_loop_u, _FP_DIV_MEAT_N_loop_v))	\
	{								\
	  _FP_FRAC_SUB_##wc (_FP_DIV_MEAT_N_loop_u,			\
			     _FP_DIV_MEAT_N_loop_u,			\
			     _FP_DIV_MEAT_N_loop_v);			\
	  _FP_FRAC_LOW_##wc (R) |= 1;					\
	  _FP_DIV_MEAT_N_loop_count--;					\
	}								\
      else								\
	R##_e--;							\
      /* Subsequent rounds.  */						\
      do								\
	{								\
	  int _FP_DIV_MEAT_N_loop_msb					\
	    = (_FP_WS_TYPE) _FP_FRAC_HIGH_##wc (_FP_DIV_MEAT_N_loop_u) < 0; \
	  _FP_FRAC_SLL_##wc (_FP_DIV_MEAT_N_loop_u, 1);			\
	  _FP_FRAC_SLL_##wc (R, 1);					\
	  if (_FP_DIV_MEAT_N_loop_msb					\
	      || _FP_FRAC_GE_1 (_FP_DIV_MEAT_N_loop_u,			\
				_FP_DIV_MEAT_N_loop_v))			\
	    {								\
	      _FP_FRAC_SUB_##wc (_FP_DIV_MEAT_N_loop_u,			\
				 _FP_DIV_MEAT_N_loop_u,			\
				 _FP_DIV_MEAT_N_loop_v);		\
	      _FP_FRAC_LOW_##wc (R) |= 1;				\
	    }								\
	}								\
      while (--_FP_DIV_MEAT_N_loop_count > 0);				\
      /* If there's anything left in _FP_DIV_MEAT_N_LOOP_U, the result	\
	 is inexact.  */						\
      _FP_FRAC_LOW_##wc (R)						\
	|= !_FP_FRAC_ZEROP_##wc (_FP_DIV_MEAT_N_loop_u);		\
    }									\
  while (0)

#define _FP_DIV_MEAT_1_loop(fs, R, X, Y)  _FP_DIV_MEAT_N_loop (fs, 1, R, X, Y)
#define _FP_DIV_MEAT_2_loop(fs, R, X, Y)  _FP_DIV_MEAT_N_loop (fs, 2, R, X, Y)
#define _FP_DIV_MEAT_4_loop(fs, R, X, Y)  _FP_DIV_MEAT_N_loop (fs, 4, R, X, Y)

#endif /* !SOFT_FP_OP_COMMON_H */
