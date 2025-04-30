/* Close a shared object opened by `_dl_open'.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <dlfcn.h>
#include <errno.h>
#include <libintl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libc-lock.h>
#include <ldsodefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sysdep-cancel.h>
#include <tls.h>
#include <stap-probe.h>

#include <dl-unmap-segments.h>


/* Type of the constructor functions.  */
typedef void (*fini_t) (void);


/* Special l_idx value used to indicate which objects remain loaded.  */
#define IDX_STILL_USED -1


/* Returns true we an non-empty was found.  */
static bool
remove_slotinfo (size_t idx, struct dtv_slotinfo_list *listp, size_t disp,
		 bool should_be_there)
{
  if (idx - disp >= listp->len)
    {
      if (listp->next == NULL)
	{
	  /* The index is not actually valid in the slotinfo list,
	     because this object was closed before it was fully set
	     up due to some error.  */
	  assert (! should_be_there);
	}
      else
	{
	  if (remove_slotinfo (idx, listp->next, disp + listp->len,
			       should_be_there))
	    return true;

	  /* No non-empty entry.  Search from the end of this element's
	     slotinfo array.  */
	  idx = disp + listp->len;
	}
    }
  else
    {
      struct link_map *old_map = listp->slotinfo[idx - disp].map;

      /* The entry might still be in its unused state if we are closing an
	 object that wasn't fully set up.  */
      if (__glibc_likely (old_map != NULL))
	{
	  /* Mark the entry as unused.  These can be read concurrently.  */
	  atomic_store_relaxed (&listp->slotinfo[idx - disp].gen,
				GL(dl_tls_generation) + 1);
	  atomic_store_relaxed (&listp->slotinfo[idx - disp].map, NULL);
	}

      /* If this is not the last currently used entry no need to look
	 further.  */
      if (idx != GL(dl_tls_max_dtv_idx))
	{
	  /* There is an unused dtv entry in the middle.  */
	  GL(dl_tls_dtv_gaps) = true;
	  return true;
	}
    }

  while (idx - disp > (disp == 0 ? 1 + GL(dl_tls_static_nelem) : 0))
    {
      --idx;

      if (listp->slotinfo[idx - disp].map != NULL)
	{
	  /* Found a new last used index.  This can be read concurrently.  */
	  atomic_store_relaxed (&GL(dl_tls_max_dtv_idx), idx);
	  return true;
	}
    }

  /* No non-entry in this list element.  */
  return false;
}

/* Invoke dstructors for CLOSURE (a struct link_map *).  Called with
   exception handling temporarily disabled, to make errors fatal.  */
static void
call_destructors (void *closure)
{
  struct link_map *map = closure;

  if (map->l_info[DT_FINI_ARRAY] != NULL)
    {
      ElfW(Addr) *array =
	(ElfW(Addr) *) (map->l_addr
			+ map->l_info[DT_FINI_ARRAY]->d_un.d_ptr);
      unsigned int sz = (map->l_info[DT_FINI_ARRAYSZ]->d_un.d_val
			 / sizeof (ElfW(Addr)));

      while (sz-- > 0)
	((fini_t) array[sz]) ();
    }

  /* Next try the old-style destructor.  */
  if (map->l_info[DT_FINI] != NULL)
    DL_CALL_DT_FINI (map, ((void *) map->l_addr
			   + map->l_info[DT_FINI]->d_un.d_ptr));
}

void
_dl_close_worker (struct link_map *map, bool force)
{
  /* One less direct use.  */
  --map->l_direct_opencount;

  /* If _dl_close is called recursively (some destructor call dlclose),
     just record that the parent _dl_close will need to do garbage collection
     again and return.  */
  static enum { not_pending, pending, rerun } dl_close_state;

  if (map->l_direct_opencount > 0 || map->l_type != lt_loaded
      || dl_close_state != not_pending)
    {
      if (map->l_direct_opencount == 0 && map->l_type == lt_loaded)
	dl_close_state = rerun;

      /* There are still references to this object.  Do nothing more.  */
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_FILES))
	_dl_debug_printf ("\nclosing file=%s; direct_opencount=%u\n",
			  map->l_name, map->l_direct_opencount);

      return;
    }

  Lmid_t nsid = map->l_ns;
  struct link_namespaces *ns = &GL(dl_ns)[nsid];

 retry:
  dl_close_state = pending;

  bool any_tls = false;
  const unsigned int nloaded = ns->_ns_nloaded;
  char used[nloaded];
  char done[nloaded];
  struct link_map *maps[nloaded];

  /* Run over the list and assign indexes to the link maps and enter
     them into the MAPS array.  */
  int idx = 0;
  for (struct link_map *l = ns->_ns_loaded; l != NULL; l = l->l_next)
    {
      /* If the first entry is the stub for when NextSilicon execve to
       * ld.so is in use, skip. */
      if (l != ns->_ns_loaded && l->l_addr == ns->_ns_loaded->l_addr)
        continue;

      l->l_idx = idx;
      maps[idx] = l;
      ++idx;

    }
  assert (idx == nloaded);

  /* Prepare the bitmaps.  */
  memset (used, '\0', sizeof (used));
  memset (done, '\0', sizeof (done));

  /* Keep track of the lowest index link map we have covered already.  */
  int done_index = -1;
  while (++done_index < nloaded)
    {
      struct link_map *l = maps[done_index];

      if (done[done_index])
	/* Already handled.  */
	continue;

      /* Check whether this object is still used.  */
      if (l->l_type == lt_loaded
	  && l->l_direct_opencount == 0
	  && !l->l_nodelete_active
	  /* See CONCURRENCY NOTES in cxa_thread_atexit_impl.c to know why
	     acquire is sufficient and correct.  */
	  && atomic_load_acquire (&l->l_tls_dtor_count) == 0
	  && !used[done_index])
	continue;

      /* We need this object and we handle it now.  */
      done[done_index] = 1;
      used[done_index] = 1;
      /* Signal the object is still needed.  */
      l->l_idx = IDX_STILL_USED;

      /* Mark all dependencies as used.  */
      if (l->l_initfini != NULL)
	{
	  /* We are always the zeroth entry, and since we don't include
	     ourselves in the dependency analysis start at 1.  */
	  struct link_map **lp = &l->l_initfini[1];
	  while (*lp != NULL)
	    {
	      if ((*lp)->l_idx != IDX_STILL_USED)
		{
		  assert ((*lp)->l_idx >= 0 && (*lp)->l_idx < nloaded);

		  if (!used[(*lp)->l_idx])
		    {
		      used[(*lp)->l_idx] = 1;
		      /* If we marked a new object as used, and we've
			 already processed it, then we need to go back
			 and process again from that point forward to
			 ensure we keep all of its dependencies also.  */
		      if ((*lp)->l_idx - 1 < done_index)
			done_index = (*lp)->l_idx - 1;
		    }
		}

	      ++lp;
	    }
	}
      /* And the same for relocation dependencies.  */
      if (l->l_reldeps != NULL)
	for (unsigned int j = 0; j < l->l_reldeps->act; ++j)
	  {
	    struct link_map *jmap = l->l_reldeps->list[j];

	    if (jmap->l_idx != IDX_STILL_USED)
	      {
		assert (jmap->l_idx >= 0 && jmap->l_idx < nloaded);

		if (!used[jmap->l_idx])
		  {
		    used[jmap->l_idx] = 1;
		    if (jmap->l_idx - 1 < done_index)
		      done_index = jmap->l_idx - 1;
		  }
	      }
	  }
    }

  /* Sort the entries.  We can skip looking for the binary itself which is
     at the front of the search list for the main namespace.  */
  _dl_sort_maps (maps + (nsid == LM_ID_BASE), nloaded - (nsid == LM_ID_BASE),
		 used + (nsid == LM_ID_BASE), true);

  /* Call all termination functions at once.  */
#ifdef SHARED
  bool do_audit = GLRO(dl_naudit) > 0 && !ns->_ns_loaded->l_auditing;
#endif
  bool unload_any = false;
  bool scope_mem_left = false;
  unsigned int unload_global = 0;
  unsigned int first_loaded = ~0;
  for (unsigned int i = 0; i < nloaded; ++i)
    {
      struct link_map *imap = maps[i];

      /* All elements must be in the same namespace.  */
      assert (imap->l_ns == nsid);

      if (!used[i])
	{
	  assert (imap->l_type == lt_loaded && !imap->l_nodelete_active);

	  /* Call its termination function.  Do not do it for
	     half-cooked objects.  Temporarily disable exception
	     handling, so that errors are fatal.  */
	  if (imap->l_init_called)
	    {
	      /* When debugging print a message first.  */
	      if (__builtin_expect (GLRO(dl_debug_mask) & DL_DEBUG_IMPCALLS,
				    0))
		_dl_debug_printf ("\ncalling fini: %s [%lu]\n\n",
				  imap->l_name, nsid);

	      if (imap->l_info[DT_FINI_ARRAY] != NULL
		  || imap->l_info[DT_FINI] != NULL)
		_dl_catch_exception (NULL, call_destructors, imap);
	    }

#ifdef SHARED
	  /* Auditing checkpoint: we remove an object.  */
	  if (__glibc_unlikely (do_audit))
	    {
	      struct audit_ifaces *afct = GLRO(dl_audit);
	      for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
		{
		  if (afct->objclose != NULL)
		    {
		      struct auditstate *state
			= link_map_audit_state (imap, cnt);
		      /* Return value is ignored.  */
		      (void) afct->objclose (&state->cookie);
		    }

		  afct = afct->next;
		}
	    }
#endif

	  /* This object must not be used anymore.  */
	  imap->l_removed = 1;

	  /* We indeed have an object to remove.  */
	  unload_any = true;

	  if (imap->l_global)
	    ++unload_global;

	  /* Remember where the first dynamically loaded object is.  */
	  if (i < first_loaded)
	    first_loaded = i;
	}
      /* Else used[i].  */
      else if (imap->l_type == lt_loaded)
	{
	  struct r_scope_elem *new_list = NULL;

	  if (imap->l_searchlist.r_list == NULL && imap->l_initfini != NULL)
	    {
	      /* The object is still used.  But one of the objects we are
		 unloading right now is responsible for loading it.  If
		 the current object does not have it's own scope yet we
		 have to create one.  This has to be done before running
		 the finalizers.

		 To do this count the number of dependencies.  */
	      unsigned int cnt;
	      for (cnt = 1; imap->l_initfini[cnt] != NULL; ++cnt)
		;

	      /* We simply reuse the l_initfini list.  */
	      imap->l_searchlist.r_list = &imap->l_initfini[cnt + 1];
	      imap->l_searchlist.r_nlist = cnt;

	      new_list = &imap->l_searchlist;
	    }

	  /* Count the number of scopes which remain after the unload.
	     When we add the local search list count it.  Always add
	     one for the terminating NULL pointer.  */
	  size_t remain = (new_list != NULL) + 1;
	  bool removed_any = false;
	  for (size_t cnt = 0; imap->l_scope[cnt] != NULL; ++cnt)
	    /* This relies on l_scope[] entries being always set either
	       to its own l_symbolic_searchlist address, or some map's
	       l_searchlist address.  */
	    if (imap->l_scope[cnt] != &imap->l_symbolic_searchlist)
	      {
		struct link_map *tmap = (struct link_map *)
		  ((char *) imap->l_scope[cnt]
		   - offsetof (struct link_map, l_searchlist));
		assert (tmap->l_ns == nsid);
		if (tmap->l_idx == IDX_STILL_USED)
		  ++remain;
		else
		  removed_any = true;
	      }
	    else
	      ++remain;

	  if (removed_any)
	    {
	      /* Always allocate a new array for the scope.  This is
		 necessary since we must be able to determine the last
		 user of the current array.  If possible use the link map's
		 memory.  */
	      size_t new_size;
	      struct r_scope_elem **newp;

#define SCOPE_ELEMS(imap) \
  (sizeof (imap->l_scope_mem) / sizeof (imap->l_scope_mem[0]))

	      if (imap->l_scope != imap->l_scope_mem
		  && remain < SCOPE_ELEMS (imap))
		{
		  new_size = SCOPE_ELEMS (imap);
		  newp = imap->l_scope_mem;
		}
	      else
		{
		  new_size = imap->l_scope_max;
		  newp = (struct r_scope_elem **)
		    malloc (new_size * sizeof (struct r_scope_elem *));
		  if (newp == NULL)
		    _dl_signal_error (ENOMEM, "dlclose", NULL,
				      N_("cannot create scope list"));
		}

	      /* Copy over the remaining scope elements.  */
	      remain = 0;
	      for (size_t cnt = 0; imap->l_scope[cnt] != NULL; ++cnt)
		{
		  if (imap->l_scope[cnt] != &imap->l_symbolic_searchlist)
		    {
		      struct link_map *tmap = (struct link_map *)
			((char *) imap->l_scope[cnt]
			 - offsetof (struct link_map, l_searchlist));
		      if (tmap->l_idx != IDX_STILL_USED)
			{
			  /* Remove the scope.  Or replace with own map's
			     scope.  */
			  if (new_list != NULL)
			    {
			      newp[remain++] = new_list;
			      new_list = NULL;
			    }
			  continue;
			}
		    }

		  newp[remain++] = imap->l_scope[cnt];
		}
	      newp[remain] = NULL;

	      struct r_scope_elem **old = imap->l_scope;

	      imap->l_scope = newp;

	      /* No user anymore, we can free it now.  */
	      if (old != imap->l_scope_mem)
		{
		  if (_dl_scope_free (old))
		    /* If _dl_scope_free used THREAD_GSCOPE_WAIT (),
		       no need to repeat it.  */
		    scope_mem_left = false;
		}
	      else
		scope_mem_left = true;

	      imap->l_scope_max = new_size;
	    }
	  else if (new_list != NULL)
	    {
	      /* We didn't change the scope array, so reset the search
		 list.  */
	      imap->l_searchlist.r_list = NULL;
	      imap->l_searchlist.r_nlist = 0;
	    }

	  /* The loader is gone, so mark the object as not having one.
	     Note: l_idx != IDX_STILL_USED -> object will be removed.  */
	  if (imap->l_loader != NULL
	      && imap->l_loader->l_idx != IDX_STILL_USED)
	    imap->l_loader = NULL;

	  /* Remember where the first dynamically loaded object is.  */
	  if (i < first_loaded)
	    first_loaded = i;
	}
    }

  /* If there are no objects to unload, do nothing further.  */
  if (!unload_any)
    goto out;

#ifdef SHARED
  /* Auditing checkpoint: we will start deleting objects.  */
  if (__glibc_unlikely (do_audit))
    {
      struct link_map *head = ns->_ns_loaded;
      struct audit_ifaces *afct = GLRO(dl_audit);
      /* Do not call the functions for any auditing object.  */
      if (head->l_auditing == 0)
	{
	  for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
	    {
	      if (afct->activity != NULL)
		{
		  struct auditstate *state = link_map_audit_state (head, cnt);
		  afct->activity (&state->cookie, LA_ACT_DELETE);
		}

	      afct = afct->next;
	    }
	}
    }
#endif

  /* Notify the debugger we are about to remove some loaded objects.  */
  struct r_debug *r = _dl_debug_initialize (0, nsid);
  r->r_state = RT_DELETE;
  _dl_debug_state ();
  LIBC_PROBE (unmap_start, 2, nsid, r);

  if (unload_global)
    {
      /* Some objects are in the global scope list.  Remove them.  */
      struct r_scope_elem *ns_msl = ns->_ns_main_searchlist;
      unsigned int i;
      unsigned int j = 0;
      unsigned int cnt = ns_msl->r_nlist;

      while (cnt > 0 && ns_msl->r_list[cnt - 1]->l_removed)
	--cnt;

      if (cnt + unload_global == ns_msl->r_nlist)
	/* Speed up removing most recently added objects.  */
	j = cnt;
      else
	for (i = 0; i < cnt; i++)
	  if (ns_msl->r_list[i]->l_removed == 0)
	    {
	      if (i != j)
		ns_msl->r_list[j] = ns_msl->r_list[i];
	      j++;
	    }
      ns_msl->r_nlist = j;
    }

  if (!RTLD_SINGLE_THREAD_P
      && (unload_global
	  || scope_mem_left
	  || (GL(dl_scope_free_list) != NULL
	      && GL(dl_scope_free_list)->count)))
    {
      THREAD_GSCOPE_WAIT ();

      /* Now we can free any queued old scopes.  */
      struct dl_scope_free_list *fsl = GL(dl_scope_free_list);
      if (fsl != NULL)
	while (fsl->count > 0)
	  free (fsl->list[--fsl->count]);
    }

  size_t tls_free_start;
  size_t tls_free_end;
  tls_free_start = tls_free_end = NO_TLS_OFFSET;

  /* We modify the list of loaded objects.  */
  __rtld_lock_lock_recursive (GL(dl_load_write_lock));

  /* Check each element of the search list to see if all references to
     it are gone.  */
  for (unsigned int i = first_loaded; i < nloaded; ++i)
    {
      struct link_map *imap = maps[i];
      if (!used[i])
	{
	  assert (imap->l_type == lt_loaded);

	  /* That was the last reference, and this was a dlopen-loaded
	     object.  We can unmap it.  */

	  /* Remove the object from the dtv slotinfo array if it uses TLS.  */
	  if (__glibc_unlikely (imap->l_tls_blocksize > 0))
	    {
	      any_tls = true;

	      if (GL(dl_tls_dtv_slotinfo_list) != NULL
		  && ! remove_slotinfo (imap->l_tls_modid,
					GL(dl_tls_dtv_slotinfo_list), 0,
					imap->l_init_called))
		/* All dynamically loaded modules with TLS are unloaded.  */
		/* Can be read concurrently.  */
		atomic_store_relaxed (&GL(dl_tls_max_dtv_idx),
				      GL(dl_tls_static_nelem));

	      if (imap->l_tls_offset != NO_TLS_OFFSET
		  && imap->l_tls_offset != FORCED_DYNAMIC_TLS_OFFSET)
		{
		  /* Collect a contiguous chunk built from the objects in
		     this search list, going in either direction.  When the
		     whole chunk is at the end of the used area then we can
		     reclaim it.  */
#if TLS_TCB_AT_TP
		  if (tls_free_start == NO_TLS_OFFSET
		      || (size_t) imap->l_tls_offset == tls_free_start)
		    {
		      /* Extend the contiguous chunk being reclaimed.  */
		      tls_free_start
			= imap->l_tls_offset - imap->l_tls_blocksize;

		      if (tls_free_end == NO_TLS_OFFSET)
			tls_free_end = imap->l_tls_offset;
		    }
		  else if (imap->l_tls_offset - imap->l_tls_blocksize
			   == tls_free_end)
		    /* Extend the chunk backwards.  */
		    tls_free_end = imap->l_tls_offset;
		  else
		    {
		      /* This isn't contiguous with the last chunk freed.
			 One of them will be leaked unless we can free
			 one block right away.  */
		      if (tls_free_end == GL(dl_tls_static_used))
			{
			  GL(dl_tls_static_used) = tls_free_start;
			  tls_free_end = imap->l_tls_offset;
			  tls_free_start
			    = tls_free_end - imap->l_tls_blocksize;
			}
		      else if ((size_t) imap->l_tls_offset
			       == GL(dl_tls_static_used))
			GL(dl_tls_static_used)
			  = imap->l_tls_offset - imap->l_tls_blocksize;
		      else if (tls_free_end < (size_t) imap->l_tls_offset)
			{
			  /* We pick the later block.  It has a chance to
			     be freed.  */
			  tls_free_end = imap->l_tls_offset;
			  tls_free_start
			    = tls_free_end - imap->l_tls_blocksize;
			}
		    }
#elif TLS_DTV_AT_TP
		  if (tls_free_start == NO_TLS_OFFSET)
		    {
		      tls_free_start = imap->l_tls_firstbyte_offset;
		      tls_free_end = (imap->l_tls_offset
				      + imap->l_tls_blocksize);
		    }
		  else if (imap->l_tls_firstbyte_offset == tls_free_end)
		    /* Extend the contiguous chunk being reclaimed.  */
		    tls_free_end = imap->l_tls_offset + imap->l_tls_blocksize;
		  else if (imap->l_tls_offset + imap->l_tls_blocksize
			   == tls_free_start)
		    /* Extend the chunk backwards.  */
		    tls_free_start = imap->l_tls_firstbyte_offset;
		  /* This isn't contiguous with the last chunk freed.
		     One of them will be leaked unless we can free
		     one block right away.  */
		  else if (imap->l_tls_offset + imap->l_tls_blocksize
			   == GL(dl_tls_static_used))
		    GL(dl_tls_static_used) = imap->l_tls_firstbyte_offset;
		  else if (tls_free_end == GL(dl_tls_static_used))
		    {
		      GL(dl_tls_static_used) = tls_free_start;
		      tls_free_start = imap->l_tls_firstbyte_offset;
		      tls_free_end = imap->l_tls_offset + imap->l_tls_blocksize;
		    }
		  else if (tls_free_end < imap->l_tls_firstbyte_offset)
		    {
		      /* We pick the later block.  It has a chance to
			 be freed.  */
		      tls_free_start = imap->l_tls_firstbyte_offset;
		      tls_free_end = imap->l_tls_offset + imap->l_tls_blocksize;
		    }
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif
		}
	    }

	  /* Reset unique symbols if forced.  */
	  if (force)
	    {
	      struct unique_sym_table *tab = &ns->_ns_unique_sym_table;
	      __rtld_lock_lock_recursive (tab->lock);
	      struct unique_sym *entries = tab->entries;
	      if (entries != NULL)
		{
		  size_t idx, size = tab->size;
		  for (idx = 0; idx < size; ++idx)
		    {
		      /* Clear unique symbol entries that belong to this
			 object.  */
		      if (entries[idx].name != NULL
			  && entries[idx].map == imap)
			{
			  entries[idx].name = NULL;
			  entries[idx].hashval = 0;
			  tab->n_elements--;
			}
		    }
		}
	      __rtld_lock_unlock_recursive (tab->lock);
	    }

	  /* We can unmap all the maps at once.  We determined the
	     start address and length when we loaded the object and
	     the `munmap' call does the rest.  */
	  DL_UNMAP (imap);

	  /* Finally, unlink the data structure and free it.  */
#if DL_NNS == 1
	  /* The assert in the (imap->l_prev == NULL) case gives
	     the compiler license to warn that NS points outside
	     the dl_ns array bounds in that case (as nsid != LM_ID_BASE
	     is tantamount to nsid >= DL_NNS).  That should be impossible
	     in this configuration, so just assert about it instead.  */
	  assert (nsid == LM_ID_BASE);
	  assert (imap->l_prev != NULL);
#else
	  if (imap->l_prev == NULL)
	    {
	      assert (nsid != LM_ID_BASE);
	      ns->_ns_loaded = imap->l_next;

	      /* Update the pointer to the head of the list
		 we leave for debuggers to examine.  */
	      r->r_map = (void *) ns->_ns_loaded;
	    }
	  else
#endif
	    imap->l_prev->l_next = imap->l_next;

	  --ns->_ns_nloaded;
	  if (imap->l_next != NULL)
	    imap->l_next->l_prev = imap->l_prev;

	  free (imap->l_versions);
	  if (imap->l_origin != (char *) -1)
	    free ((char *) imap->l_origin);

	  free (imap->l_reldeps);

	  /* Print debugging message.  */
	  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_FILES))
	    _dl_debug_printf ("\nfile=%s [%lu];  destroying link map\n",
			      imap->l_name, imap->l_ns);

	  /* This name always is allocated.  */
	  free (imap->l_name);
	  /* Remove the list with all the names of the shared object.  */

	  struct libname_list *lnp = imap->l_libname;
	  do
	    {
	      struct libname_list *this = lnp;
	      lnp = lnp->next;
	      if (!this->dont_free)
		free (this);
	    }
	  while (lnp != NULL);

	  /* Remove the searchlists.  */
	  free (imap->l_initfini);

	  /* Remove the scope array if we allocated it.  */
	  if (imap->l_scope != imap->l_scope_mem)
	    free (imap->l_scope);

	  if (imap->l_phdr_allocated)
	    free ((void *) imap->l_phdr);

	  if (imap->l_rpath_dirs.dirs != (void *) -1)
	    free (imap->l_rpath_dirs.dirs);
	  if (imap->l_runpath_dirs.dirs != (void *) -1)
	    free (imap->l_runpath_dirs.dirs);

	  /* Clear GL(dl_initfirst) when freeing its link_map memory.  */
	  if (imap == GL(dl_initfirst))
	    GL(dl_initfirst) = NULL;

	  free (imap);
	}
    }

  __rtld_lock_unlock_recursive (GL(dl_load_write_lock));

  /* If we removed any object which uses TLS bump the generation counter.  */
  if (any_tls)
    {
      size_t newgen = GL(dl_tls_generation) + 1;
      if (__glibc_unlikely (newgen == 0))
	_dl_fatal_printf ("TLS generation counter wrapped!  Please report as described in "REPORT_BUGS_TO".\n");
      /* Can be read concurrently.  */
      atomic_store_relaxed (&GL(dl_tls_generation), newgen);

      if (tls_free_end == GL(dl_tls_static_used))
	GL(dl_tls_static_used) = tls_free_start;
    }

#ifdef SHARED
  /* Auditing checkpoint: we have deleted all objects.  */
  if (__glibc_unlikely (do_audit))
    {
      struct link_map *head = ns->_ns_loaded;
      /* If head is NULL, the namespace has become empty, and the
	 audit interface does not give us a way to signal
	 LA_ACT_CONSISTENT for it because the first loaded module is
	 used to identify the namespace.

	 Furthermore, do not notify auditors of the cleanup of a
	 failed audit module loading attempt.  */
      if (head != NULL && head->l_auditing == 0)
	{
	  struct audit_ifaces *afct = GLRO(dl_audit);
	  for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
	    {
	      if (afct->activity != NULL)
		{
		  struct auditstate *state = link_map_audit_state (head, cnt);
		  afct->activity (&state->cookie, LA_ACT_CONSISTENT);
		}

	      afct = afct->next;
	    }
	}
    }
#endif

  if (__builtin_expect (ns->_ns_loaded == NULL, 0)
      && nsid == GL(dl_nns) - 1)
    do
      --GL(dl_nns);
    while (GL(dl_ns)[GL(dl_nns) - 1]._ns_loaded == NULL);

  /* Notify the debugger those objects are finalized and gone.  */
  r->r_state = RT_CONSISTENT;
  _dl_debug_state ();
  LIBC_PROBE (unmap_complete, 2, nsid, r);

  /* Recheck if we need to retry, release the lock.  */
 out:
  if (dl_close_state == rerun)
    goto retry;

  dl_close_state = not_pending;
}


void
_dl_close (void *_map)
{
#if !DISABLE_DLCLOSE
  struct link_map *map = _map;

  /* We must take the lock to examine the contents of map and avoid
     concurrent dlopens.  */
  __rtld_lock_lock_recursive (GL(dl_load_lock));

  /* At this point we are guaranteed nobody else is touching the list of
     loaded maps, but a concurrent dlclose might have freed our map
     before we took the lock. There is no way to detect this (see below)
     so we proceed assuming this isn't the case.  First see whether we
     can remove the object at all.  */
  if (__glibc_unlikely (map->l_nodelete_active))
    {
      /* Nope.  Do nothing.  */
      __rtld_lock_unlock_recursive (GL(dl_load_lock));
      return;
    }

  /* At present this is an unreliable check except in the case where the
     caller has recursively called dlclose and we are sure the link map
     has not been freed.  In a non-recursive dlclose the map itself
     might have been freed and this access is potentially a data race
     with whatever other use this memory might have now, or worse we
     might silently corrupt memory if it looks enough like a link map.
     POSIX has language in dlclose that appears to guarantee that this
     should be a detectable case and given that dlclose should be threadsafe
     we need this to be a reliable detection.
     This is bug 20990. */
  if (__builtin_expect (map->l_direct_opencount, 1) == 0)
    {
      __rtld_lock_unlock_recursive (GL(dl_load_lock));
      _dl_signal_error (0, map->l_name, NULL, N_("shared object not open"));
    }

  _dl_close_worker (map, false);

  __rtld_lock_unlock_recursive (GL(dl_load_lock));
#endif
}
