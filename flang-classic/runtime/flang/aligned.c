/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
/** \file
 *  Routines to test array conformance
 */
#include "fort_vars.h"

/** \brief
 * Compare rank and shape of objects s and t. Return true if s and t
 * are conformable under the axis mappings smap and tmap. Scalars are
 * conformable with arrays. conform(A,B) implies conform(B,A).
*/

int I8(__fort_conform)(F90_Desc *s, __INT_T *smap, F90_Desc *t, __INT_T *tmap)
{
  DECL_DIM_PTRS(sd);
  DECL_DIM_PTRS(td);
  __INT_T dim;

  if (s == NULL || t == NULL)
    return 0;
  if (s == t || F90_TAG_G(s) != __DESC || F90_TAG_G(t) != __DESC)
    return 1;
  if (F90_GSIZE_G(s) == 0 && F90_GSIZE_G(t) == 0)
    return 1;
  if (F90_RANK_G(s) != F90_RANK_G(t))
    return 0;
  for (dim = F90_RANK_G(s); --dim >= 0;) {
    SET_DIM_PTRS(sd, s, smap[dim] - 1);
    SET_DIM_PTRS(td, t, tmap[dim] - 1);
    if (F90_DPTR_EXTENT_G(sd) != F90_DPTR_EXTENT_G(sd))
      return 0;
  }
  return 1;
}

int I8(__fort_covers_procs)(F90_Desc *s, F90_Desc *t)
{
  return 1;
}


/* TODO: delete? */

/** \brief
 * Leftover from HPF Fortran
 */
int I8(__fort_aligned)(F90_Desc *t, __INT_T *tmap, F90_Desc *u, __INT_T *umap)
{
  return 1;
}

/* Same as aligned(), except examine only the corresponding axes tx
   and ux, and assume that t's processor set is covered by u's
   processor set. */

/* TODO: delete? */

/** \brief
 * Leftover from HPF Fortran
 */
int I8(__fort_aligned_axes)(F90_Desc *t, int tx, F90_Desc *u, int ux)
{
  return 1;
}

