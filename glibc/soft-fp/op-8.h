/* Software floating-point emulation.
   Basic eight-word fraction declaration and manipulation.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com),
		  Jakub Jelinek (jj@ultra.linux.cz) and
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

#ifndef SOFT_FP_OP_8_H
#define SOFT_FP_OP_8_H	1

/* We need just a few things from here for op-4, if we ever need some
   other macros, they can be added.  */
#define _FP_FRAC_DECL_8(X)	_FP_W_TYPE X##_f[8]
#define _FP_FRAC_SET_8(X, I)    __FP_FRAC_SET_8 (X, I)
#define _FP_FRAC_HIGH_8(X)	(X##_f[7])
#define _FP_FRAC_LOW_8(X)	(X##_f[0])
#define _FP_FRAC_WORD_8(X, w)	(X##_f[w])

#define _FP_FRAC_SLL_8(X, N)						\
  do									\
    {									\
      _FP_I_TYPE _FP_FRAC_SLL_8_up, _FP_FRAC_SLL_8_down;		\
      _FP_I_TYPE _FP_FRAC_SLL_8_skip, _FP_FRAC_SLL_8_i;			\
      _FP_FRAC_SLL_8_skip = (N) / _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SLL_8_up = (N) % _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SLL_8_down = _FP_W_TYPE_SIZE - _FP_FRAC_SLL_8_up;	\
      if (!_FP_FRAC_SLL_8_up)						\
	for (_FP_FRAC_SLL_8_i = 7;					\
	     _FP_FRAC_SLL_8_i >= _FP_FRAC_SLL_8_skip;			\
	     --_FP_FRAC_SLL_8_i)					\
	  X##_f[_FP_FRAC_SLL_8_i]					\
	    = X##_f[_FP_FRAC_SLL_8_i-_FP_FRAC_SLL_8_skip];		\
      else								\
	{								\
	  for (_FP_FRAC_SLL_8_i = 7;					\
	       _FP_FRAC_SLL_8_i > _FP_FRAC_SLL_8_skip;			\
	       --_FP_FRAC_SLL_8_i)					\
	    X##_f[_FP_FRAC_SLL_8_i]					\
	      = ((X##_f[_FP_FRAC_SLL_8_i-_FP_FRAC_SLL_8_skip]		\
		  << _FP_FRAC_SLL_8_up)					\
		 | (X##_f[_FP_FRAC_SLL_8_i-_FP_FRAC_SLL_8_skip-1]	\
		    >> _FP_FRAC_SLL_8_down));				\
	  X##_f[_FP_FRAC_SLL_8_i--] = X##_f[0] << _FP_FRAC_SLL_8_up;	\
	}								\
      for (; _FP_FRAC_SLL_8_i >= 0; --_FP_FRAC_SLL_8_i)			\
	X##_f[_FP_FRAC_SLL_8_i] = 0;					\
    }									\
  while (0)

#define _FP_FRAC_SRL_8(X, N)						\
  do									\
    {									\
      _FP_I_TYPE _FP_FRAC_SRL_8_up, _FP_FRAC_SRL_8_down;		\
      _FP_I_TYPE _FP_FRAC_SRL_8_skip, _FP_FRAC_SRL_8_i;			\
      _FP_FRAC_SRL_8_skip = (N) / _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SRL_8_down = (N) % _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SRL_8_up = _FP_W_TYPE_SIZE - _FP_FRAC_SRL_8_down;	\
      if (!_FP_FRAC_SRL_8_down)						\
	for (_FP_FRAC_SRL_8_i = 0;					\
	     _FP_FRAC_SRL_8_i <= 7-_FP_FRAC_SRL_8_skip;			\
	     ++_FP_FRAC_SRL_8_i)					\
	  X##_f[_FP_FRAC_SRL_8_i]					\
	    = X##_f[_FP_FRAC_SRL_8_i+_FP_FRAC_SRL_8_skip];		\
      else								\
	{								\
	  for (_FP_FRAC_SRL_8_i = 0;					\
	       _FP_FRAC_SRL_8_i < 7-_FP_FRAC_SRL_8_skip;		\
	       ++_FP_FRAC_SRL_8_i)					\
	    X##_f[_FP_FRAC_SRL_8_i]					\
	      = ((X##_f[_FP_FRAC_SRL_8_i+_FP_FRAC_SRL_8_skip]		\
		  >> _FP_FRAC_SRL_8_down)				\
		 | (X##_f[_FP_FRAC_SRL_8_i+_FP_FRAC_SRL_8_skip+1]	\
		    << _FP_FRAC_SRL_8_up));				\
	  X##_f[_FP_FRAC_SRL_8_i++] = X##_f[7] >> _FP_FRAC_SRL_8_down;	\
	}								\
      for (; _FP_FRAC_SRL_8_i < 8; ++_FP_FRAC_SRL_8_i)			\
	X##_f[_FP_FRAC_SRL_8_i] = 0;					\
    }									\
  while (0)


/* Right shift with sticky-lsb.
   What this actually means is that we do a standard right-shift,
   but that if any of the bits that fall off the right hand side
   were one then we always set the LSbit.  */
#define _FP_FRAC_SRS_8(X, N, size)					\
  do									\
    {									\
      _FP_I_TYPE _FP_FRAC_SRS_8_up, _FP_FRAC_SRS_8_down;		\
      _FP_I_TYPE _FP_FRAC_SRS_8_skip, _FP_FRAC_SRS_8_i;			\
      _FP_W_TYPE _FP_FRAC_SRS_8_s;					\
      _FP_FRAC_SRS_8_skip = (N) / _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SRS_8_down = (N) % _FP_W_TYPE_SIZE;			\
      _FP_FRAC_SRS_8_up = _FP_W_TYPE_SIZE - _FP_FRAC_SRS_8_down;	\
      for (_FP_FRAC_SRS_8_s = _FP_FRAC_SRS_8_i = 0;			\
	   _FP_FRAC_SRS_8_i < _FP_FRAC_SRS_8_skip;			\
	   ++_FP_FRAC_SRS_8_i)						\
	_FP_FRAC_SRS_8_s |= X##_f[_FP_FRAC_SRS_8_i];			\
      if (!_FP_FRAC_SRS_8_down)						\
	for (_FP_FRAC_SRS_8_i = 0;					\
	     _FP_FRAC_SRS_8_i <= 7-_FP_FRAC_SRS_8_skip;			\
	     ++_FP_FRAC_SRS_8_i)					\
	  X##_f[_FP_FRAC_SRS_8_i]					\
	    = X##_f[_FP_FRAC_SRS_8_i+_FP_FRAC_SRS_8_skip];		\
      else								\
	{								\
	  _FP_FRAC_SRS_8_s						\
	    |= X##_f[_FP_FRAC_SRS_8_i] << _FP_FRAC_SRS_8_up;		\
	  for (_FP_FRAC_SRS_8_i = 0;					\
	       _FP_FRAC_SRS_8_i < 7-_FP_FRAC_SRS_8_skip;		\
	       ++_FP_FRAC_SRS_8_i)					\
	    X##_f[_FP_FRAC_SRS_8_i]					\
	      = ((X##_f[_FP_FRAC_SRS_8_i+_FP_FRAC_SRS_8_skip]		\
		  >> _FP_FRAC_SRS_8_down)				\
		 | (X##_f[_FP_FRAC_SRS_8_i+_FP_FRAC_SRS_8_skip+1]	\
		    << _FP_FRAC_SRS_8_up));				\
	  X##_f[_FP_FRAC_SRS_8_i++] = X##_f[7] >> _FP_FRAC_SRS_8_down;	\
	}								\
      for (; _FP_FRAC_SRS_8_i < 8; ++_FP_FRAC_SRS_8_i)			\
	X##_f[_FP_FRAC_SRS_8_i] = 0;					\
      /* Don't fix the LSB until the very end when we're sure f[0] is	\
	 stable.  */							\
      X##_f[0] |= (_FP_FRAC_SRS_8_s != 0);				\
    }									\
  while (0)

#define _FP_FRAC_ADD_8(R, X, Y)                                             \
  do                                                                        \
    {                                                                       \
      _FP_W_TYPE _FP_FRAC_ADD_8_c = 0;                                      \
      _FP_I_TYPE _FP_FRAC_ADD_8_i;                                          \
      for (_FP_FRAC_ADD_8_i = 0; _FP_FRAC_ADD_8_i < 8; ++_FP_FRAC_ADD_8_i)  \
        {                                                                   \
          R##_f[_FP_FRAC_ADD_8_i]                                           \
            = (X##_f[_FP_FRAC_ADD_8_i] + Y##_f[_FP_FRAC_ADD_8_i]            \
               + _FP_FRAC_ADD_8_c);                                         \
          _FP_FRAC_ADD_8_c                                                  \
            = (_FP_FRAC_ADD_8_c                                             \
               ? R##_f[_FP_FRAC_ADD_8_i] <= X##_f[_FP_FRAC_ADD_8_i]         \
               : R##_f[_FP_FRAC_ADD_8_i] < X##_f[_FP_FRAC_ADD_8_i]);        \
        }                                                                   \
    }                                                                       \
  while (0)

#define _FP_FRAC_SUB_8(R, X, Y)                                             \
  do                                                                        \
    {                                                                       \
      _FP_W_TYPE _FP_FRAC_SUB_8_tmp[8];                                     \
      _FP_W_TYPE _FP_FRAC_SUB_8_c = 0;                                      \
      _FP_I_TYPE _FP_FRAC_SUB_8_i;                                          \
      for (_FP_FRAC_SUB_8_i = 0; _FP_FRAC_SUB_8_i < 8; ++_FP_FRAC_SUB_8_i)  \
        {                                                                   \
          _FP_FRAC_SUB_8_tmp[_FP_FRAC_SUB_8_i]                              \
            = (X##_f[_FP_FRAC_SUB_8_i] - Y##_f[_FP_FRAC_SUB_8_i]            \
               - _FP_FRAC_SUB_8_c);                                         \
          _FP_FRAC_SUB_8_c                                                  \
            = (_FP_FRAC_SUB_8_c                                             \
               ? (_FP_FRAC_SUB_8_tmp[_FP_FRAC_SUB_8_i]                      \
                  >= X##_f[_FP_FRAC_SUB_8_i])                               \
               : (_FP_FRAC_SUB_8_tmp[_FP_FRAC_SUB_8_i]                      \
                  > X##_f[_FP_FRAC_SUB_8_i]));                              \
        }                                                                   \
      for (_FP_FRAC_SUB_8_i = 0; _FP_FRAC_SUB_8_i < 8; ++_FP_FRAC_SUB_8_i)  \
        R##_f[_FP_FRAC_SUB_8_i] = _FP_FRAC_SUB_8_tmp[_FP_FRAC_SUB_8_i];     \
    }                                                                       \
  while (0)

#define _FP_FRAC_CLZ_8(R, X)                                                \
  do                                                                        \
    {                                                                       \
      _FP_I_TYPE _FP_FRAC_CLZ_8_i;                                          \
      for (_FP_FRAC_CLZ_8_i = 7; _FP_FRAC_CLZ_8_i > 0; _FP_FRAC_CLZ_8_i--)  \
        if (X##_f[_FP_FRAC_CLZ_8_i])                                        \
          break;                                                            \
      __FP_CLZ ((R), X##_f[_FP_FRAC_CLZ_8_i]);                              \
      (R) += _FP_W_TYPE_SIZE * (7 - _FP_FRAC_CLZ_8_i);                      \
    }                                                                       \
  while (0)

#define _FP_MINFRAC_8   0, 0, 0, 0, 0, 0, 0, 1

#define _FP_FRAC_NEGP_8(X)      ((_FP_WS_TYPE) X##_f[7] < 0)
#define _FP_FRAC_ZEROP_8(X)                                             \
  ((X##_f[0] | X##_f[1] | X##_f[2] | X##_f[3]                           \
    | X##_f[4] | X##_f[5] | X##_f[6] | X##_f[7]) == 0)
#define _FP_FRAC_HIGHBIT_DW_8(fs, X)                                    \
  (_FP_FRAC_HIGH_DW_##fs (X) & _FP_HIGHBIT_DW_##fs)

#define _FP_FRAC_COPY_4_8(D, S)                           \
  do                                                      \
    {                                                     \
      D##_f[0] = S##_f[0];                                \
      D##_f[1] = S##_f[1];                                \
      D##_f[2] = S##_f[2];                                \
      D##_f[3] = S##_f[3];                                \
    }                                                     \
  while (0)

#define _FP_FRAC_COPY_8_4(D, S)                           \
  do                                                      \
    {                                                     \
      D##_f[0] = S##_f[0];                                \
      D##_f[1] = S##_f[1];                                \
      D##_f[2] = S##_f[2];                                \
      D##_f[3] = S##_f[3];                                \
      D##_f[4] = D##_f[5] = D##_f[6] = D##_f[7]= 0;       \
    }                                                     \
  while (0)

#define __FP_FRAC_SET_8(X, I7, I6, I5, I4, I3, I2, I1, I0)             \
  (X##_f[7] = I7, X##_f[6] = I6, X##_f[5] = I5, X##_f[4] = I4,         \
   X##_f[3] = I3, X##_f[2] = I2, X##_f[1] = I1, X##_f[0] = I0)

#endif /* !SOFT_FP_OP_8_H */
