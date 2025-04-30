/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* FIXME: is this necessary for Flang?*/

/** \file
 * These routines are substituted for intrinsic procedures which are
 * passed as actual arguments.  The intrinsics don't expect
 * descriptors -- these routines do.
 */

#include <string.h>
#include "fioMacros.h"
/* macros for entries */
#if defined(_WIN64)
#define ENTFTN_MS I8
#endif

/*
      integer function ENTFTN(len)(string)
      character*(*) string
*/

__INT_T
ENTFTN(LENA, lena)
(DCHAR(string) DCLEN64(string))
{
  return (__INT_T)CLEN(string);
}

/* 32 bit CLEN version */
__INT_T
ENTFTN(LEN, len)
(DCHAR(string) DCLEN(string))
{
  return ENTFTN(LENA, lena)
  (CADR(string), (__CLEN_T)CLEN(string));
}

/* Version of ENTFTN(len) that never takes a descriptor. */
__INT_T
ENTFTN(LENXA, lenxa)(DCHAR(string) DCLEN64(string)) { return (__INT_T)CLEN(string); }
/* 32 bit CLEN version */
__INT_T
ENTFTN(LENX, lenx)(DCHAR(string) DCLEN(string)) { return (__INT_T)CLEN(string); }

__INT8_T
ENTFTN(KLENA, klena)
(DCHAR(string) DCLEN64(string))
{
  return (__INT8_T)CLEN(string);
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KLEN, klen)
(DCHAR(string) DCLEN(string))
{
  return ENTFTN(KLENA, klena)
    (CADR(string), (__CLEN_T)CLEN(string));
}

/* Version of ENTFTN(klenx) that never takes a descriptor. */
__INT8_T
ENTFTN(KLENXA, klenxa)(DCHAR(string) DCLEN64(string)) { return (__INT8_T)CLEN(string); }
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KLENX, klenx)(DCHAR(string) DCLEN(string)) { return (__INT8_T)CLEN(string); }

/*
      Per the standard, the procedural version of index does not accept
      the back argument:

      integer function ENTFTN(index)(string, substring)
      character*(*) string, substring
*/
__INT_T
ENTFTN(INDEXA, indexa)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  __INT_T i, n;

  n = (__INT_T)CLEN(string) - (__INT_T)CLEN(substring);
  if (n < 0)
    return 0;

  if (CLEN(substring) == 0)
    return 1;
  for (i = 0; i <= n; ++i) {
    if (CADR(string)[i] == CADR(substring)[0] &&
        strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
      return i + 1;
  }
  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTFTN(INDEX, index)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return ENTFTN(INDEXA, indexa)
    (CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

/** \brief version of index that takes no descriptor */
__INT_T
ENTFTN(INDEXXA, indexxa)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(INDEX, index)(CADR(string), CADR(substring),
                              CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT_T
ENTFTN(INDEXX, indexx)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return ENTFTN(INDEXXA, indexxa)(CADR(string), CADR(substring),
              (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

__INT8_T
ENTFTN(KINDEXA, kindexa)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  __INT8_T i, n;

  n = (__INT8_T)CLEN(string) - (__INT8_T)CLEN(substring);
  if (n < 0)
    return 0;

  if (CLEN(substring) == 0)
    return 1;
  for (i = 0; i <= n; ++i) {
    if (CADR(string)[i] == CADR(substring)[0] &&
        strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
      return i + 1;
  }
  return 0;
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KINDEX, kindex)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return (__INT8_T)ENTFTN(KINDEXA, kindexa)
    (CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

/** \brief version of index that takes no descriptor */
__INT8_T
ENTFTN(KINDEXXA, kindexxa)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(KINDEXA, kindexa)(CADR(string), CADR(substring),
                                CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KINDEXX, kindexx)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return ENTFTN(KINDEXXA, kindexxa)(CADR(string), CADR(substring),
                          (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

#if defined(TARGET_WIN)

/** \brief Version of pg_len that never takes a descriptor.
 * For cref , mixedstrlen
 */
__INT_T
ENTFTN(LENXA, lenx_cra)(DCHAR(string) DCLEN64(string)) { return (__INT_T)CLEN(string); }
/* 32 bit CLEN version */
__INT_T
ENTFTN(LENX, lenx_cr)(DCHAR(string) DCLEN(string))
{ 
  return ENTFTN(LENXA, lenx_cra)(CADR(string), (__CLEN_T)CLEN(string));
}

/* For cref , nomixedstrlen*/
__INT_T
ENTFTN(LENXA, lenx_cr_nma)(DCHAR(string) DCLEN64(string)) { return (__INT_T)CLEN(string); }
/* 32 bit CLEN version */
__INT_T
ENTFTN(LENX, lenx_cr_nm)(DCHAR(string) DCLEN(string))
{
  return ENTFTN(LENXA, lenx_cr_nma)(CADR(string), (__CLEN_T)CLEN(string));
}

/** \brief for cref, mixedstrlen*/
__INT8_T
ENTFTN(KLENXA, klenx_cra)(DCHAR(string) DCLEN64(string)) { return (__INT8_T)CLEN(string); }
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KLENX, klenx_cr)(DCHAR(string) DCLEN(string))
{
  return ENTFTN(KLENXA, klenx_cra)(CADR(string), (__CLEN_T)CLEN(string));
}

/* for cref, nomixedstrlen*/
__INT8_T
ENTFTN(KLENXA, klenx_cr_nma)(DCHAR(string) DCLEN64(string)) { return (__INT8_T)CLEN(string); }
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KLENX, klenx_cr_nm)(DCHAR(string) DCLEN(string))
{
  return ENTFTN(KLENXA, klenx_cr_nma)(CADR(string), (__CLEN_T)CLEN(string));
}

/* For cref, mixed strlen*/
__INT_T
ENTFTN(INDEXXA, indexx_cra)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  return ENTFTN(INDEXA, indexa)(CADR(string), CADR(substring),
                              CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT_T
ENTFTN(INDEXX, indexx_cr)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN(INDEXX, indexx_cr)(CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief For cref, nomixed strlen*/
__INT_T
ENTFTN(INDEXXA, indexx_cr_nma)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(INDEXA, indexa)(CADR(string), CADR(substring),
                              CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT_T
ENTFTN(INDEXX, indexx_cr_nm)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return ENTFTN(INDEXXA, indexx_cr_nma)(CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

/** \brief* for cref, mixedstrlen */
__INT8_T
ENTFTN(KINDEXXA, kindexx_cra)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  return ENTFTN(KINDEXA, kindexa)(CADR(string), CADR(substring),
                                CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KINDEXX, kindexx_cr)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN(KINDEXXA, kindexx_cra)(CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief* for cref, nomixedstrlen */
__INT8_T
ENTFTN(KINDEXXA, kindexx_cr_nma)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(KINDEXA, kindexa)(CADR(string), CADR(substring),
                                CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN(KINDEXX, kindexx_cr_nm)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return (__INT8_T) ENTFTN(KINDEXXA, kindexx_cr_nma) (CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

#endif

#if defined(_WIN64)

/* functions here follow the msfortran/mscall conventions */

__INT_T
ENTFTN_MS(PGHPF_LENA)
(DCHAR(string) DCLEN64(string))
{
  return (__INT_T)CLEN(string);
}
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_LEN)
(DCHAR(string) DCLEN(string))
{
  return ENTFTN_MS(PGHPF_LENA)
    (CADR(string), (__CLEN_T)CLEN(string));
}

/** \brief Version of pg_len that never takes a descriptor.
 * This is necessary for pghpf -Mf90
 */
__INT_T
ENTFTN_MS(PGHPF_LENXA)(DCHAR(string) DCLEN64(string)) { return (__INT_T)CLEN(string); }
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_LENX)(DCHAR(string) DCLEN(string))
{
  return ENTFTN_MS(PGHPF_LENXA)(CADR(string), (__CLEN_T)CLEN(string));
}

/* Version of pg_len that never takes a descriptor.
 * This is necessary for pghpf -Mf90, -Miface=nomixed_str_len_arg */
__INT_T
ENTFTN_MS(PGHPF_LENX_NMA)(DCHAR(string) DCLEN64(string)) { return (__INT_T)CLEN(string); }
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_LENX_NM)(DCHAR(string) DCLEN(string))
{
  return (__INT_T)ENTFTN_MS(PGHPF_LENX_NMA)(CADR(string), (__CLEN_T)CLEN(string));
}

__INT8_T
ENTFTN_MS(PGHPF_KLENA)
(DCHAR(string) DCLEN64(string))
{
  return (__INT8_T)CLEN(string);
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KLEN)
(DCHAR(string) DCLEN(string))
{
  return ENTFTN_MS(PGHPF_KLENA)
    (CADR(string), (__CLEN_T)CLEN(string));
}

/* Version of pg_lenx that never takes a descriptor.
 * This is necessary for pghpf -Mf90 */
__INT8_T
ENTFTN_MS(PGHPF_KLENXA)(DCHAR(string) DCLEN64(string)) { return (__INT8_T)CLEN(string); }
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KLENX)(DCHAR(string) DCLEN(string))
{
  return ENTFTN_MS(PGHPF_KLENXA)(CADR(string), (__CLEN_T)CLEN(string));
}

/** \brief Version of lenx that never takes a descriptor.  */
__INT8_T
ENTFTN_MS(PGHPF_KLENX_NMA)(DCHAR(string) DCLEN64(string)) { return (__INT8_T)CLEN(string); }
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KLENX_NM)(DCHAR(string) DCLEN(string))
{
  return ENTFTN_MS(PGHPF_KLENX_NMA)(CADR(string), (__CLEN_T)CLEN(string));
}

/*
      Per the standard, the procedural version of index does not accept
      the back argument:

      integer function index(string, substring)
      character*(*) string, substring
*/
/*   pghpf versions are passed descriptors; pgf90 versions are not. */
__INT_T
ENTFTN_MS(PGHPF_INDEXA)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  __INT_T i, n;

  n = (__INT_T)CLEN(string) - (__INT_T)CLEN(substring);
  if (n < 0)
    return 0;

  if (CLEN(substring) == 0)
    return 1;
  for (i = 0; i <= n; ++i) {
    if (CADR(string)[i] == CADR(substring)[0] &&
        strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
      return i + 1;
  }
  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_INDEX)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN_MS(PGHPF_INDEXA)
    (CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief version of index that takes no descriptor, used in pghpf
 * -Mf90
 */
__INT_T
ENTFTN_MS(PGHPF_INDEXXA)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  return ENTFTN(INDEXA, indexa)(CADR(string), CADR(substring),
                              CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_INDEXX)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN_MS(PGHPF_INDEXXA) (CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief Version of index that takes no descriptor, used in pghp 
 * -Miface=nomixedstrlen
 */
__INT_T
ENTFTN_MS(PGHPF_INDEXX_NMA)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(INDEXA, indexa)(CADR(string), CADR(substring),
                              CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT_T
ENTFTN_MS(PGHPF_INDEXX_NM)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
return ENTFTN_MS(PGHPF_INDEXX_NMA) (CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

__INT8_T
ENTFTN_MS(PGHPF_KINDEXA)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  __INT8_T i, n;

  n = (__INT8_T)CLEN(string) - (__INT8_T)CLEN(substring);
  if (n < 0)
    return 0;

  if (CLEN(substring) == 0)
    return 1;
  for (i = 0; i <= n; ++i) {
    if (CADR(string)[i] == CADR(substring)[0] &&
        strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
      return i + 1;
  }
  return 0;
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KINDEX)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN_MS(PGHPF_KINDEXA)
    (CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief version of index that takes no descriptor, used in pghpf
 * -Mf90 */
__INT8_T
ENTFTN_MS(PGHPF_KINDEXXA)
(DCHAR(string) DCLEN64(string), DCHAR(substring) DCLEN64(substring))
{
  return ENTFTN(KINDEXA, kindexa)(CADR(string), CADR(substring),
                                CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KINDEXX)
(DCHAR(string) DCLEN(string), DCHAR(substring) DCLEN(substring))
{
  return ENTFTN_MS(PGHPF_KINDEXXA)
    (CADR(string), (__CLEN_T)CLEN(string), CADR(substring), (__CLEN_T)CLEN(substring));
}

/** \brief version of index that takes no descriptor,
 * -Miface=nomixedstrlen used in pghpf 
 */
__INT8_T
ENTFTN_MS(PGHPF_KINDEXX_NMA)
(DCHAR(string), DCHAR(substring) DCLEN64(string) DCLEN64(substring))
{
  return ENTFTN(KINDEXA, kindexa)(CADR(string), CADR(substring),
                                CLEN(string), CLEN(substring));
}
/* 32 bit CLEN version */
__INT8_T
ENTFTN_MS(PGHPF_KINDEXX_NM)
(DCHAR(string), DCHAR(substring) DCLEN(string) DCLEN(substring))
{
  return ENTFTN_MS(PGHPF_KINDEXX_NMA)
    (CADR(string), CADR(substring), (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

#endif
