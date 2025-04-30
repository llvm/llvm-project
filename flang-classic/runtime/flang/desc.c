/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* desc.c - Fortran90 descriptor IO */

#include "global.h"
#include "descRW.h"
#include "feddesc.h"
#include "format.h"
#include "unf.h"
#include "list_io.h"

__INT_T
ENTFTNIO(FMT_READ, fmt_read)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 0, __f90io_fmt_read);
}

__INT_T
ENTFTNIO(FMT_WRITE, fmt_write)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 1, __f90io_fmt_write);
}

__INT_T
ENTFTNIO(LDR, ldr)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 0, __f90io_ldr);
}

__INT_T
ENTFTNIO(LDW, ldw)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 1, __f90io_ldw);
}

__INT_T
ENTFTNIO(USW_READ, usw_read)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 0, __f90io_usw_read);
}

__INT_T
ENTFTNIO(USW_WRITE, usw_write)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 1, __f90io_usw_write);
}

__INT_T
ENTFTNIO(UNF_READ, unf_read)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 0, __f90io_unf_read);
}

__INT_T
ENTFTNIO(UNF_WRITE, unf_write)
(char *ab,     /* base address */
 F90_Desc *ac) /* descriptor */
{
  return I8(__fortio_main)(ab, ac, 1, __f90io_unf_write);
}

#define TEMPLATE(dd, i, lb, ub, lbase, gsize)                                  \
  {                                                                            \
    __INT_T __extent, u, l;                                                    \
    DECL_DIM_PTRS(_dd);                                                        \
    SET_DIM_PTRS(_dd, dd, i - 1);                                              \
    l = lb;                                                                    \
    u = ub;                                                                    \
    __extent = u - l + 1;                                                      \
    if (__extent < 0) {                                                        \
      __extent = 0;                                                            \
      u = l - 1;                                                               \
    }                                                                          \
    F90_DPTR_LBOUND_P(_dd, l);                                                 \
    DPTR_UBOUND_P(_dd, u);                                                     \
    F90_DPTR_SSTRIDE_P(_dd, 1);                                                \
    F90_DPTR_SOFFSET_P(_dd, 0);                                                \
    lbase -= gsize * l;                                                        \
    F90_DPTR_LSTRIDE_P(_dd, gsize);                                            \
    gsize *= __extent;                                                         \
  }

void
I8(get_vlist_desc)(F90_Desc *sd, __INT_T ubnd)
{
  __INT_T rank = 1;
  __INT_T flags = 0;
  __INT_T lbnd = 1;
  __INT_T gsize, lbase;
  int type;
#ifdef DESC_I8
  type = __INT8;
#else
  type = __INT4;
#endif

  __DIST_INIT_DESCRIPTOR(sd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  F90_LBASE_P(sd, lbnd);
  TEMPLATE(sd, 1, lbnd, ubnd, lbase, gsize);
  F90_LBASE_P(sd, lbase);
  F90_LSIZE_P(sd, gsize);
  F90_GSIZE_P(sd, gsize);
  F90_KIND_P(sd, type);
  F90_LEN_P(sd, sizeof(__INT_T));
}
