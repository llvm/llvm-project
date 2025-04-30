/* Hash table for TLS descriptors.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Alexandre Oliva  <aoliva@redhat.com>

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

#ifndef TLSDESCHTAB_H
# define TLSDESCHTAB_H 1

#include <atomic.h>

# ifdef SHARED

#  include <inline-hashtab.h>

inline static int
hash_tlsdesc (void *p)
{
  struct tlsdesc_dynamic_arg *td = p;

  /* We know all entries are for the same module, so ti_offset is the
     only distinguishing entry.  */
  return td->tlsinfo.ti_offset;
}

inline static int
eq_tlsdesc (void *p, void *q)
{
  struct tlsdesc_dynamic_arg *tdp = p, *tdq = q;

  return tdp->tlsinfo.ti_offset == tdq->tlsinfo.ti_offset;
}

inline static size_t
map_generation (struct link_map *map)
{
  size_t idx = map->l_tls_modid;
  struct dtv_slotinfo_list *listp = GL(dl_tls_dtv_slotinfo_list);

  /* Find the place in the dtv slotinfo list.  */
  do
    {
      /* Does it fit in the array of this list element?  */
      if (idx < listp->len)
	{
	  /* We should never get here for a module in static TLS, so
	     we can assume that, if the generation count is zero, we
	     still haven't determined the generation count for this
	     module.  */
	  if (listp->slotinfo[idx].map == map && listp->slotinfo[idx].gen)
	    return listp->slotinfo[idx].gen;
	  else
	    break;
	}
      idx -= listp->len;
      listp = listp->next;
    }
  while (listp != NULL);

  /* If we get to this point, the module still hasn't been assigned an
     entry in the dtv slotinfo data structures, and it will when we're
     done with relocations.  At that point, the module will get a
     generation number that is one past the current generation, so
     return exactly that.  */
  return GL(dl_tls_generation) + 1;
}

/* Returns the data pointer for a given map and tls offset that is used
   to fill in one of the GOT entries referenced by a TLSDESC relocation
   when using dynamic TLS.  This requires allocation, returns NULL on
   allocation failure.  */
void *
_dl_make_tlsdesc_dynamic (struct link_map *map, size_t ti_offset)
{
  struct hashtab *ht;
  void **entry;
  struct tlsdesc_dynamic_arg *td, test;

  ht = map->l_mach.tlsdesc_table;
  if (! ht)
    {
      ht = htab_create ();
      if (! ht)
	return 0;
      map->l_mach.tlsdesc_table = ht;
    }

  test.tlsinfo.ti_module = map->l_tls_modid;
  test.tlsinfo.ti_offset = ti_offset;
  entry = htab_find_slot (ht, &test, 1, hash_tlsdesc, eq_tlsdesc);
  if (! entry)
    return 0;

  if (*entry)
    {
      td = *entry;
      return td;
    }

  *entry = td = malloc (sizeof (struct tlsdesc_dynamic_arg));
  /* This may be higher than the map's generation, but it doesn't
     matter much.  Worst case, we'll have one extra DTV update per
     thread.  */
  td->gen_count = map_generation (map);
  td->tlsinfo = test.tlsinfo;
  return td;
}

# endif /* SHARED */

#endif
