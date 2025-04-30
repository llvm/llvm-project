/* Common definitions for libm tests for vector functions.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define TEST_MATHVEC 1
#define TEST_NARROW 0
#define TEST_ERRNO 0
#define TEST_EXCEPTIONS 0

#define CNCT(x, y) x ## y
#define CONCAT(a, b) CNCT (a, b)

#define WRAPPER_NAME(function) CONCAT (function, VEC_SUFF)
#define FUNC_TEST(function) WRAPPER_NAME (FUNC (function))

/* This macro is used in VECTOR_WRAPPER macros for vector tests.  */
#define TEST_VEC_LOOP(vec, len) 				\
  do								\
    {								\
      for (i = 1; i < len; i++)					\
        {							\
          if ((FLOAT) vec[0] != (FLOAT) vec[i])			\
            {							\
              vec[0] = (FLOAT) vec[0] + 0.1;			\
	      break;						\
            }							\
        }							\
    }								\
  while (0)

#define INIT_VEC_LOOP(vec, val, len)				\
  do								\
    {								\
      for (i = 0; i < len; i++)					\
        {							\
          vec[i] = val;						\
        }							\
    }								\
  while (0)

#define WRAPPER_DECL_f(function) extern FLOAT function (FLOAT);
#define WRAPPER_DECL_ff(function) extern FLOAT function (FLOAT, FLOAT);
#define WRAPPER_DECL_fFF(function) extern void function (FLOAT, FLOAT *, FLOAT *);

/* Wrapper from scalar to vector function.  */
#define VECTOR_WRAPPER(scalar_func, vector_func) \
extern VEC_TYPE vector_func (VEC_TYPE);		\
FLOAT scalar_func (FLOAT x)			\
{						\
  int i;					\
  VEC_TYPE mx;					\
  INIT_VEC_LOOP (mx, x, VEC_LEN);		\
  VEC_TYPE mr = vector_func (mx);		\
  TEST_VEC_LOOP (mr, VEC_LEN);			\
  return ((FLOAT) mr[0]);			\
}

/* Wrapper from scalar 2 argument function to vector one.  */
#define VECTOR_WRAPPER_ff(scalar_func, vector_func) 	\
extern VEC_TYPE vector_func (VEC_TYPE, VEC_TYPE);	\
FLOAT scalar_func (FLOAT x, FLOAT y)		\
{						\
  int i;					\
  VEC_TYPE mx, my;				\
  INIT_VEC_LOOP (mx, x, VEC_LEN);		\
  INIT_VEC_LOOP (my, y, VEC_LEN);		\
  VEC_TYPE mr = vector_func (mx, my);		\
  TEST_VEC_LOOP (mr, VEC_LEN);			\
  return ((FLOAT) mr[0]);			\
}

/* Wrapper from scalar 3 argument function to vector one.  */
#define VECTOR_WRAPPER_fFF(scalar_func, vector_func) 	\
extern void vector_func (VEC_TYPE, VEC_TYPE *, VEC_TYPE *);	\
void scalar_func (FLOAT x, FLOAT * r, FLOAT * r1)		\
{						\
  int i;					\
  VEC_TYPE mx, mr, mr1;				\
  INIT_VEC_LOOP (mx, x, VEC_LEN);		\
  vector_func (mx, &mr, &mr1);			\
  TEST_VEC_LOOP (mr, VEC_LEN);			\
  TEST_VEC_LOOP (mr1, VEC_LEN);			\
  *r = (FLOAT) mr[0];				\
  *r1 = (FLOAT) mr1[0];				\
  return;					\
}
