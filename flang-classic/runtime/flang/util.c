/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* utility functions */

#include <stdioInterf.h>
#include "fioMacros.h"

/* given address and size, fetch scalar integer and convert to C int */

int I8(__fort_varying_int)(void *b, __INT_T *size)
{
  switch (*size) {
  case 1:
    return (int)(*(__INT1_T *)b);
  case 2:
    return (int)(*(__INT2_T *)b);
  case 4:
    return (int)(*(__INT4_T *)b);
  case 8:
    return (int)(*(__INT8_T *)b);
  default:
    __fort_abort("varying_int: incorrect size");
    return 0;
  }
}

/* given address and size, fetch scalar logical and convert to C int */

int I8(__fort_varying_log)(void *b, __INT_T *size)
{
  switch (*size) {
  case 1:
    return (*(__LOG1_T *)b & GET_DIST_MASK_LOG1) != 0;
  case 2:
    return (*(__LOG2_T *)b & GET_DIST_MASK_LOG2) != 0;
  case 4:
    return (*(__LOG4_T *)b & GET_DIST_MASK_LOG4) != 0;
  case 8:
    return (*(__LOG8_T *)b & GET_DIST_MASK_LOG8) != 0;
  default:
    __fort_abort("varying_log: incorrect size");
    return 0;
  }
}

/* given address and descriptor, fetch scalar integer and convert to C
   int */

int I8(__fort_fetch_int)(void *b, F90_Desc *d)
{
  dtype kind;

  if (F90_TAG_G(d) == __DESC) {
    if (F90_RANK_G(d) != 0)
      __fort_abort("fetch_int: non-scalar destination");
    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      __fort_abort("fetch_int: non-local value");
    b += DIST_SCOFF_G(d) * F90_LEN_G(d);
    kind = F90_KIND_G(d);
  } else
    kind = Abs(F90_TAG_G(d));

  switch (kind) {
  case __INT1:
    return (int)(*(__INT1_T *)b);
  case __INT2:
    return (int)(*(__INT2_T *)b);
  case __INT4:
    return (int)(*(__INT4_T *)b);
  case __INT8:
    return (int)(*(__INT8_T *)b);
  default:
    __fort_abort("fetch_int: non-integer type");
    return 0;
  }
}

/* store scalar integer */

void I8(__fort_store_int)(void *b, F90_Desc *d, int val)
{
  dtype kind;

  if (F90_TAG_G(d) == __DESC) {
    if (F90_RANK_G(d) != 0)
      __fort_abort("store_int: non-scalar destination");
    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      return;
    b += DIST_SCOFF_G(d) * F90_LEN_G(d);
    kind = F90_KIND_G(d);
  } else
    kind = Abs(F90_TAG_G(d));

  switch (kind) {
  case __INT1:
    *(__INT1_T *)b = (__INT1_T)val;
    break;
  case __INT2:
    *(__INT2_T *)b = (__INT2_T)val;
    break;
  case __INT4:
    *(__INT4_T *)b = (__INT4_T)val;
    break;
  case __INT8:
    *(__INT8_T *)b = (__INT8_T)val;
    break;
  default:
    __fort_abort("store_int: non-integer type");
  }
}

/* given address and descriptor, fetch scalar fortran logical and
   convert to C int */

int I8(__fort_fetch_log)(void *b, F90_Desc *d)
{
  dtype kind;

  if (F90_TAG_G(d) == __DESC) {
    if (F90_RANK_G(d) != 0)
      __fort_abort("fetch_log: non-scalar destination");
    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      __fort_abort("fetch_int: non-local value");
    b += DIST_SCOFF_G(d) * F90_LEN_G(d);
    kind = F90_KIND_G(d);
  } else
    kind = Abs(F90_TAG_G(d));

  switch (kind) {
  case __LOG1:
    return (*(__LOG1_T *)b & GET_DIST_MASK_LOG1) != 0;
  case __LOG2:
    return (*(__LOG2_T *)b & GET_DIST_MASK_LOG2) != 0;
  case __LOG4:
    return (*(__LOG4_T *)b & GET_DIST_MASK_LOG4) != 0;
  case __LOG8:
    return (*(__LOG8_T *)b & GET_DIST_MASK_LOG8) != 0;
  default:
    __fort_abort("fetch_log: non-logical type");
    return 0;
  }
}

/* convert C int and store scalar fortran logical */

void I8(__fort_store_log)(void *b, F90_Desc *d, int val)
{
  dtype kind;

  if (F90_TAG_G(d) == __DESC) {
    if (F90_RANK_G(d) != 0)
      __fort_abort("store_log: non-scalar destination");
    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      return;
    b += DIST_SCOFF_G(d) * F90_LEN_G(d);
    kind = F90_KIND_G(d);
  } else
    kind = Abs(F90_TAG_G(d));

  switch (kind) {
  case __LOG1:
    *(__LOG1_T *)b = val ? GET_DIST_TRUE_LOG1 : 0;
    break;
  case __LOG2:
    *(__LOG2_T *)b = val ? GET_DIST_TRUE_LOG2 : 0;
    break;
  case __LOG4:
    *(__LOG4_T *)b = val ? GET_DIST_TRUE_LOG4 : 0;
    break;
  case __LOG8:
    *(__LOG8_T *)b = val ? GET_DIST_TRUE_LOG8 : 0;
    break;
  default:
    __fort_abort("store_log: non-logical type");
  }
}

/* fetch the i'th element of an integer vector */

int I8(__fort_fetch_int_element)(void *b, F90_Desc *d, int i)
{
  double tmp[2];
  __INT_T idx;
  int val = 0;

  if (F90_RANK_G(d) != 1)
    __fort_abort("fetch_int_element: non-unit rank");

  idx = F90_DIM_LBOUND_G(d, 0) - 1 + i;
  I8(__fort_get_scalar)(tmp, b, d, &idx);
  switch (F90_KIND_G(d)) {
  case __INT1:
    val = (int)(*(__INT1_T *)tmp);
    break;
  case __INT2:
    val = (int)(*(__INT2_T *)tmp);
    break;
  case __INT4:
    val = (int)(*(__INT4_T *)tmp);
    break;
  case __INT8:
    val = (int)(*(__INT8_T *)tmp);
    break;
  default:
    __fort_abort("fetch_int_element: non-integer type");
  }
  return val;
}

/* store the i'th element of an integer vector */

void I8(__fort_store_int_element)(void *b, F90_Desc *d, int i, int val)
{
  void *adr;
  __INT_T idx;

  if (F90_RANK_G(d) != 1)
    __fort_abort("store_int_element: non-unit rank");

  idx = F90_DIM_LBOUND_G(d, 0) - 1 + i;
  adr = I8(__fort_local_address)(b, d, &idx);
  if (adr != NULL) {
    switch (F90_KIND_G(d)) {
    case __INT1:
      *(__INT1_T *)adr = (__INT1_T)val;
      break;
    case __INT2:
      *(__INT2_T *)adr = (__INT2_T)val;
      break;
    case __INT4:
      *(__INT4_T *)adr = (__INT4_T)val;
      break;
    case __INT8:
      *(__INT8_T *)adr = (__INT8_T)val;
      break;
    default:
      __fort_abort("store_int_element: non-integer type");
    }
  }
}

/* fetch integer vector */

void I8(__fort_fetch_int_vector)(void *b, F90_Desc *d, int *vec, int veclen)
{
  double tmp[2];
  __INT_T i;

  if (F90_RANK_G(d) != 1)
    __fort_abort("fetch_vector: non-unit rank");

  for (i = F90_DIM_LBOUND_G(d, 0); --veclen >= 0; ++i, ++vec) {
    I8(__fort_get_scalar)(tmp, b, d, &i);
    switch (F90_KIND_G(d)) {
    case __INT1:
      *vec = (int)(*(__INT1_T *)tmp);
      break;
    case __INT2:
      *vec = (int)(*(__INT2_T *)tmp);
      break;
    case __INT4:
      *vec = (int)(*(__INT4_T *)tmp);
      break;
    case __INT8:
      *vec = (int)(*(__INT8_T *)tmp);
      break;
    default:
      __fort_abort("fetch_int_vector: non-integer type");
    }
  }
}

/* store integer vector */

void I8(__fort_store_int_vector)(void *b, F90_Desc *d, int *vec, int veclen)
{
  void *adr;
  __INT_T i;

  if (F90_RANK_G(d) != 1)
    __fort_abort("store_int_vector: non-unit rank");

  for (i = F90_DIM_LBOUND_G(d, 0); --veclen >= 0; ++i, ++vec) {
    adr = I8(__fort_local_address)(b, d, &i);
    if (adr != NULL) {
      switch (F90_KIND_G(d)) {
      case __INT1:
        *(__INT1_T *)adr = (__INT1_T)*vec;
        break;
      case __INT2:
        *(__INT2_T *)adr = (__INT2_T)*vec;
        break;
      case __INT4:
        *(__INT4_T *)adr = (__INT4_T)*vec;
        break;
      case __INT8:
        *(__INT8_T *)adr = (__INT8_T)*vec;
        break;
      default:
        __fort_abort("store_int_vector: non-integer type");
      }
    }
  }
}

#ifndef DESC_I8

/* fortran string copy */
void __fort_ftnstrcpy(char *dst, /*  destination string, blank-filled */
                     int len,   /*  length of destination space */
                     char *src) /*  null terminated source string  */
{
  char *end = dst + len;
  while (dst < end && *src != '\0')
    *dst++ = *src++;
  while (dst < end)
    *dst++ = ' ';
}

#endif
