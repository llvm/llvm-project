/* Call the termination functions of loaded shared objects.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <string.h>
#include <ldsodefs.h>
#include <elf-initfini.h>


/* Type of the constructor functions.  */
typedef void (*fini_t) (void);


void
_dl_fini (void)
{
  /* Lots of fun ahead.  We have to call the destructors for all still
     loaded objects, in all namespaces.  The problem is that the ELF
     specification now demands that dependencies between the modules
     are taken into account.  I.e., the destructor for a module is
     called before the ones for any of its dependencies.

     To make things more complicated, we cannot simply use the reverse
     order of the constructors.  Since the user might have loaded objects
     using `dlopen' there are possibly several other modules with its
     dependencies to be taken into account.  Therefore we have to start
     determining the order of the modules once again from the beginning.  */

  /* We run the destructors of the main namespaces last.  As for the
     other namespaces, we pick run the destructors in them in reverse
     order of the namespace ID.  */
#ifdef SHARED
  int do_audit = 0;
 again:
#endif
  for (Lmid_t ns = GL(dl_nns) - 1; ns >= 0; --ns)
    {
      /* Protect against concurrent loads and unloads.  */
      __rtld_lock_lock_recursive (GL(dl_load_lock));

      unsigned int nloaded = GL(dl_ns)[ns]._ns_nloaded;
      /* No need to do anything for empty namespaces or those used for
	 auditing DSOs.  */
      if (nloaded == 0
#ifdef SHARED
	  || GL(dl_ns)[ns]._ns_loaded->l_auditing != do_audit
#endif
	  )
	__rtld_lock_unlock_recursive (GL(dl_load_lock));
      else
	{
	  /* Now we can allocate an array to hold all the pointers and
	     copy the pointers in.  */
	  struct link_map *maps[nloaded];

	  unsigned int i;
	  struct link_map *l;
	  assert (nloaded != 0 || GL(dl_ns)[ns]._ns_loaded == NULL);
	  for (l = GL(dl_ns)[ns]._ns_loaded, i = 0; l != NULL; l = l->l_next)
	  {
            /* If the first entry is the stub for when NextSilicon execve to
             * ld.so is in use, skip. */
            if (l != GL(dl_ns)[ns]._ns_loaded &&
                l->l_addr == GL(dl_ns)[ns]._ns_loaded->l_addr)
              continue;

	    /* Do not handle ld.so in secondary namespaces.  */
	    if (l == l->l_real)
	      {
		assert (i < nloaded);

		maps[i] = l;
		l->l_idx = i;
		++i;

		/* Bump l_direct_opencount of all objects so that they
		   are not dlclose()ed from underneath us.  */
		++l->l_direct_opencount;
	      }
	  }
	  assert (ns != LM_ID_BASE || i == nloaded);
	  assert (ns == LM_ID_BASE || i == nloaded || i == nloaded - 1);
	  unsigned int nmaps = i;

	  /* Now we have to do the sorting.  We can skip looking for the
	     binary itself which is at the front of the search list for
	     the main namespace.  */
	  _dl_sort_maps (maps + (ns == LM_ID_BASE), nmaps - (ns == LM_ID_BASE),
			 NULL, true);

	  /* We do not rely on the linked list of loaded object anymore
	     from this point on.  We have our own list here (maps).  The
	     various members of this list cannot vanish since the open
	     count is too high and will be decremented in this loop.  So
	     we release the lock so that some code which might be called
	     from a destructor can directly or indirectly access the
	     lock.  */
	  __rtld_lock_unlock_recursive (GL(dl_load_lock));

	  /* 'maps' now contains the objects in the right order.  Now
	     call the destructors.  We have to process this array from
	     the front.  */
	  for (i = 0; i < nmaps; ++i)
	    {
	      struct link_map *l = maps[i];

	      if (l->l_init_called)
		{
		  /* Make sure nothing happens if we are called twice.  */
		  l->l_init_called = 0;

		  /* Is there a destructor function?  */
		  if (l->l_info[DT_FINI_ARRAY] != NULL
		      || (ELF_INITFINI && l->l_info[DT_FINI] != NULL))
		    {
		      /* When debugging print a message first.  */
		      if (__builtin_expect (GLRO(dl_debug_mask)
					    & DL_DEBUG_IMPCALLS, 0))
			_dl_debug_printf ("\ncalling fini: %s [%lu]\n\n",
					  DSO_FILENAME (l->l_name),
					  ns);

		      /* First see whether an array is given.  */
		      if (l->l_info[DT_FINI_ARRAY] != NULL)
			{
			  ElfW(Addr) *array =
			    (ElfW(Addr) *) (l->l_addr
					    + l->l_info[DT_FINI_ARRAY]->d_un.d_ptr);
			  unsigned int i = (l->l_info[DT_FINI_ARRAYSZ]->d_un.d_val
					    / sizeof (ElfW(Addr)));
			  while (i-- > 0)
			    ((fini_t) array[i]) ();
			}

		      /* Next try the old-style destructor.  */
		      if (ELF_INITFINI && l->l_info[DT_FINI] != NULL)
			DL_CALL_DT_FINI
			  (l, l->l_addr + l->l_info[DT_FINI]->d_un.d_ptr);
		    }

#ifdef SHARED
		  /* Auditing checkpoint: another object closed.  */
		  if (!do_audit && __builtin_expect (GLRO(dl_naudit) > 0, 0))
		    {
		      struct audit_ifaces *afct = GLRO(dl_audit);
		      for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
			{
			  if (afct->objclose != NULL)
			    {
			      struct auditstate *state
				= link_map_audit_state (l, cnt);
			      /* Return value is ignored.  */
			      (void) afct->objclose (&state->cookie);
			    }
			  afct = afct->next;
			}
		    }
#endif
		}

	      /* Correct the previous increment.  */
	      --l->l_direct_opencount;
	    }
	}
    }

#ifdef SHARED
  if (! do_audit && GLRO(dl_naudit) > 0)
    {
      do_audit = 1;
      goto again;
    }

  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_STATISTICS))
    _dl_debug_printf ("\nruntime linker statistics:\n"
		      "           final number of relocations: %lu\n"
		      "final number of relocations from cache: %lu\n",
		      GL(dl_num_relocations),
		      GL(dl_num_cache_relocations));
#endif
}
