/* Sort array of link maps according to dependencies.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <ldsodefs.h>


/* Sort array MAPS according to dependencies of the contained objects.
   Array USED, if non-NULL, is permutated along MAPS.  If FOR_FINI this is
   called for finishing an object.  */
void
_dl_sort_maps (struct link_map **maps, unsigned int nmaps, char *used,
	       bool for_fini)
{
  /* A list of one element need not be sorted.  */
  if (nmaps <= 1)
    return;

  unsigned int i = 0;
  uint16_t seen[nmaps];
  memset (seen, 0, nmaps * sizeof (seen[0]));
  while (1)
    {
      /* Keep track of which object we looked at this round.  */
      ++seen[i];
      struct link_map *thisp = maps[i];

      if (__glibc_unlikely (for_fini))
	{
	  /* Do not handle ld.so in secondary namespaces and objects which
	     are not removed.  */
	  if (thisp != thisp->l_real || thisp->l_idx == -1)
	    goto skip;
	}

      /* Find the last object in the list for which the current one is
	 a dependency and move the current object behind the object
	 with the dependency.  */
      unsigned int k = nmaps - 1;
      while (k > i)
	{
	  struct link_map **runp = maps[k]->l_initfini;
	  if (runp != NULL)
	    /* Look through the dependencies of the object.  */
	    while (*runp != NULL)
	      if (__glibc_unlikely (*runp++ == thisp))
		{
		move:
		  /* Move the current object to the back past the last
		     object with it as the dependency.  */
		  memmove (&maps[i], &maps[i + 1],
			   (k - i) * sizeof (maps[0]));
		  maps[k] = thisp;

		  if (used != NULL)
		    {
		      char here_used = used[i];
		      memmove (&used[i], &used[i + 1],
			       (k - i) * sizeof (used[0]));
		      used[k] = here_used;
		    }

		  if (seen[i + 1] > nmaps - i)
		    {
		      ++i;
		      goto next_clear;
		    }

		  uint16_t this_seen = seen[i];
		  memmove (&seen[i], &seen[i + 1], (k - i) * sizeof (seen[0]));
		  seen[k] = this_seen;

		  goto next;
		}

	  if (__glibc_unlikely (for_fini && maps[k]->l_reldeps != NULL))
	    {
	      unsigned int m = maps[k]->l_reldeps->act;
	      struct link_map **relmaps = &maps[k]->l_reldeps->list[0];

	      /* Look through the relocation dependencies of the object.  */
	      while (m-- > 0)
		if (__glibc_unlikely (relmaps[m] == thisp))
		  {
		    /* If a cycle exists with a link time dependency,
		       preserve the latter.  */
		    struct link_map **runp = thisp->l_initfini;
		    if (runp != NULL)
		      while (*runp != NULL)
			if (__glibc_unlikely (*runp++ == maps[k]))
			  goto ignore;
		    goto move;
		  }
	    ignore:;
	    }

	  --k;
	}

    skip:
      if (++i == nmaps)
	break;
    next_clear:
      memset (&seen[i], 0, (nmaps - i) * sizeof (seen[0]));

    next:;
    }
}
