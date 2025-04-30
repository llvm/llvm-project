/* Memory handling for the scope data structures.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <ldsodefs.h>
#include <sysdep-cancel.h>


int
_dl_scope_free (void *old)
{
  struct dl_scope_free_list *fsl;
#define DL_SCOPE_FREE_LIST_SIZE (sizeof (fsl->list) / sizeof (fsl->list[0]))

  if (RTLD_SINGLE_THREAD_P)
    free (old);
  else if ((fsl = GL(dl_scope_free_list)) == NULL)
    {
      GL(dl_scope_free_list) = fsl = malloc (sizeof (*fsl));
      if (fsl == NULL)
	{
	  THREAD_GSCOPE_WAIT ();
	  free (old);
	  return 1;
	}
      else
	{
	  fsl->list[0] = old;
	  fsl->count = 1;
	}
    }
  else if (fsl->count < DL_SCOPE_FREE_LIST_SIZE)
    fsl->list[fsl->count++] = old;
  else
    {
      THREAD_GSCOPE_WAIT ();
      while (fsl->count > 0)
	free (fsl->list[--fsl->count]);
      return 1;
    }
  return 0;
}
