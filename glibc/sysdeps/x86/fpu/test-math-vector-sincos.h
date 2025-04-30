/* Wrappers definitions for tests of ABI of vector sincos/sincosf having
   vector declaration "#pragma omp declare simd notinbranch".
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#define INIT_VEC_PTRS_LOOP(vec, val, len)				\
  do									\
    {									\
      union { VEC_INT_TYPE v; __typeof__ ((val)[0]) *a[(len)]; } u;	\
      for (i = 0; i < len; i++)						\
	u.a[i] = &(val)[i];						\
      (vec) = u.v;							\
    }									\
  while (0)

/* Wrapper for vector sincos/sincosf compatible with x86_64 and x32 variants
   of _ZGVbN2vvv_sincos, _ZGVdN4vvv_sincos, _ZGVeN8vvv_sincos;
   x32 variants of _ZGVbN4vvv_sincosf, _ZGVcN4vvv_sincos, _ZGVdN8vvv_sincosf,
   _ZGVeN16vvv_sincosf.  */
#define VECTOR_WRAPPER_fFF_2(scalar_func, vector_func)		\
extern void vector_func (VEC_TYPE, VEC_INT_TYPE, VEC_INT_TYPE);	\
void scalar_func (FLOAT x, FLOAT * r, FLOAT * r1)		\
{								\
  int i;							\
  FLOAT r_loc[VEC_LEN], r1_loc[VEC_LEN];			\
  VEC_TYPE mx;							\
  VEC_INT_TYPE mr, mr1;						\
  INIT_VEC_LOOP (mx, x, VEC_LEN);				\
  INIT_VEC_PTRS_LOOP (mr, r_loc, VEC_LEN);			\
  INIT_VEC_PTRS_LOOP (mr1, r1_loc, VEC_LEN);			\
  vector_func (mx, mr, mr1);					\
  TEST_VEC_LOOP (r_loc, VEC_LEN);				\
  TEST_VEC_LOOP (r1_loc, VEC_LEN);				\
  *r = r_loc[0];						\
  *r1 = r1_loc[0];						\
  return;							\
}

/* Wrapper for vector sincos/sincosf compatible with x86_64 variants of
   _ZGVcN4vvv_sincos, _ZGVeN16vvv_sincosf, _ZGVbN4vvv_sincosf,
   _ZGVdN8vvv_sincosf, _ZGVcN8vvv_sincosf.  */
#define VECTOR_WRAPPER_fFF_3(scalar_func, vector_func)		\
extern void vector_func (VEC_TYPE, VEC_INT_TYPE, VEC_INT_TYPE,  \
			 VEC_INT_TYPE, VEC_INT_TYPE);		\
void scalar_func (FLOAT x, FLOAT * r, FLOAT * r1)		\
{								\
  int i;							\
  FLOAT r_loc[VEC_LEN/2], r1_loc[VEC_LEN/2];			\
  VEC_TYPE mx;							\
  VEC_INT_TYPE mr, mr1;						\
  INIT_VEC_LOOP (mx, x, VEC_LEN);				\
  INIT_VEC_PTRS_LOOP (mr, r_loc, VEC_LEN/2);			\
  INIT_VEC_PTRS_LOOP (mr1, r1_loc, VEC_LEN/2);			\
  vector_func (mx, mr, mr, mr1, mr1);				\
  TEST_VEC_LOOP (r_loc, VEC_LEN/2);				\
  TEST_VEC_LOOP (r1_loc, VEC_LEN/2);				\
  *r = r_loc[0];						\
  *r1 = r1_loc[0];						\
  return;							\
}

/* Wrapper for vector sincosf compatible with x86_64 variant of
   _ZGVcN8vvv_sincosf.  */
#define VECTOR_WRAPPER_fFF_4(scalar_func, vector_func) 		\
extern void vector_func (VEC_TYPE, VEC_INT_TYPE, VEC_INT_TYPE,	\
			 VEC_INT_TYPE, VEC_INT_TYPE,		\
			 VEC_INT_TYPE, VEC_INT_TYPE,		\
			 VEC_INT_TYPE, VEC_INT_TYPE);		\
void scalar_func (FLOAT x, FLOAT * r, FLOAT * r1)		\
{								\
  int i;							\
  FLOAT r_loc[VEC_LEN/4], r1_loc[VEC_LEN/4];			\
  VEC_TYPE mx;							\
  VEC_INT_TYPE mr, mr1;						\
  INIT_VEC_LOOP (mx, x, VEC_LEN);				\
  INIT_VEC_PTRS_LOOP (mr, r_loc, VEC_LEN/4);			\
  INIT_VEC_PTRS_LOOP (mr1, r1_loc, VEC_LEN/4);			\
  vector_func (mx, mr, mr, mr, mr, mr1, mr1, mr1, mr1);		\
  TEST_VEC_LOOP (r_loc, VEC_LEN/4);				\
  TEST_VEC_LOOP (r1_loc, VEC_LEN/4);				\
  *r = r_loc[0];						\
  *r1 = r1_loc[0];						\
  return;							\
}
