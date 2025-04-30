/* Load a shared object at runtime, relocate it, and run its initializer.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>		/* Check whether MAP_COPY is defined.  */
#include <sys/param.h>
#include <libc-lock.h>
#include <ldsodefs.h>
#include <sysdep-cancel.h>
#include <tls.h>
#include <stap-probe.h>
#include <atomic.h>
#include <libc-internal.h>
#include <array_length.h>
#include <libc-early-init.h>
#include <gnu/lib-names.h>

#include <dl-dst.h>
#include <dl-prop.h>


/* We must be careful not to leave us in an inconsistent state.  Thus we
   catch any error and re-raise it after cleaning up.  */

struct dl_open_args
{
  const char *file;
  int mode;
  /* This is the caller of the dlopen() function.  */
  const void *caller_dlopen;
  struct link_map *map;
  /* Namespace ID.  */
  Lmid_t nsid;

  /* Original value of _ns_global_scope_pending_adds.  Set by
     dl_open_worker.  Only valid if nsid is a real namespace
     (non-negative).  */
  unsigned int original_global_scope_pending_adds;

  /* Set to true by dl_open_worker if libc.so was already loaded into
     the namespace at the time dl_open_worker was called.  This is
     used to determine whether libc.so early initialization has
     already been done before, and whether to roll back the cached
     libc_map value in the namespace in case of a dlopen failure.  */
  bool libc_already_loaded;

  /* Original parameters to the program and the current environment.  */
  int argc;
  char **argv;
  char **env;
};

/* Called in case the global scope cannot be extended.  */
static void __attribute__ ((noreturn))
add_to_global_resize_failure (struct link_map *new)
{
  _dl_signal_error (ENOMEM, new->l_libname->name, NULL,
		    N_ ("cannot extend global scope"));
}

/* Grow the global scope array for the namespace, so that all the new
   global objects can be added later in add_to_global_update, without
   risk of memory allocation failure.  add_to_global_resize raises
   exceptions for memory allocation errors.  */
static void
add_to_global_resize (struct link_map *new)
{
  struct link_namespaces *ns = &GL (dl_ns)[new->l_ns];

  /* Count the objects we have to put in the global scope.  */
  unsigned int to_add = 0;
  for (unsigned int cnt = 0; cnt < new->l_searchlist.r_nlist; ++cnt)
    if (new->l_searchlist.r_list[cnt]->l_global == 0)
      ++to_add;

  /* The symbols of the new objects and its dependencies are to be
     introduced into the global scope that will be used to resolve
     references from other dynamically-loaded objects.

     The global scope is the searchlist in the main link map.  We
     extend this list if necessary.  There is one problem though:
     since this structure was allocated very early (before the libc
     is loaded) the memory it uses is allocated by the malloc()-stub
     in the ld.so.  When we come here these functions are not used
     anymore.  Instead the malloc() implementation of the libc is
     used.  But this means the block from the main map cannot be used
     in an realloc() call.  Therefore we allocate a completely new
     array the first time we have to add something to the locale scope.  */

  if (__builtin_add_overflow (ns->_ns_global_scope_pending_adds, to_add,
			      &ns->_ns_global_scope_pending_adds))
    add_to_global_resize_failure (new);

  unsigned int new_size = 0; /* 0 means no new allocation.  */
  void *old_global = NULL; /* Old allocation if free-able.  */

  /* Minimum required element count for resizing.  Adjusted below for
     an exponential resizing policy.  */
  size_t required_new_size;
  if (__builtin_add_overflow (ns->_ns_main_searchlist->r_nlist,
			      ns->_ns_global_scope_pending_adds,
			      &required_new_size))
    add_to_global_resize_failure (new);

  if (ns->_ns_global_scope_alloc == 0)
    {
      if (__builtin_add_overflow (required_new_size, 8, &new_size))
	add_to_global_resize_failure (new);
    }
  else if (required_new_size > ns->_ns_global_scope_alloc)
    {
      if (__builtin_mul_overflow (required_new_size, 2, &new_size))
	add_to_global_resize_failure (new);

      /* The old array was allocated with our malloc, not the minimal
	 malloc.  */
      old_global = ns->_ns_main_searchlist->r_list;
    }

  if (new_size > 0)
    {
      size_t allocation_size;
      if (__builtin_mul_overflow (new_size, sizeof (struct link_map *),
				  &allocation_size))
	add_to_global_resize_failure (new);
      struct link_map **new_global = malloc (allocation_size);
      if (new_global == NULL)
	add_to_global_resize_failure (new);

      /* Copy over the old entries.  */
      memcpy (new_global, ns->_ns_main_searchlist->r_list,
	      ns->_ns_main_searchlist->r_nlist * sizeof (struct link_map *));

      ns->_ns_global_scope_alloc = new_size;
      ns->_ns_main_searchlist->r_list = new_global;

      if (!RTLD_SINGLE_THREAD_P)
	THREAD_GSCOPE_WAIT ();

      free (old_global);
    }
}

/* Actually add the new global objects to the global scope.  Must be
   called after add_to_global_resize.  This function cannot fail.  */
static void
add_to_global_update (struct link_map *new)
{
  struct link_namespaces *ns = &GL (dl_ns)[new->l_ns];

  /* Now add the new entries.  */
  unsigned int new_nlist = ns->_ns_main_searchlist->r_nlist;
  for (unsigned int cnt = 0; cnt < new->l_searchlist.r_nlist; ++cnt)
    {
      struct link_map *map = new->l_searchlist.r_list[cnt];

      if (map->l_global == 0)
	{
	  map->l_global = 1;

	  /* The array has been resized by add_to_global_resize.  */
	  assert (new_nlist < ns->_ns_global_scope_alloc);

	  ns->_ns_main_searchlist->r_list[new_nlist++] = map;

	  /* We modify the global scope.  Report this.  */
	  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_SCOPES))
	    _dl_debug_printf ("\nadd %s [%lu] to global scope\n",
			      map->l_name, map->l_ns);
	}
    }

  /* Some of the pending adds have been performed by the loop above.
     Adjust the counter accordingly.  */
  unsigned int added = new_nlist - ns->_ns_main_searchlist->r_nlist;
  assert (added <= ns->_ns_global_scope_pending_adds);
  ns->_ns_global_scope_pending_adds -= added;

  atomic_write_barrier ();
  ns->_ns_main_searchlist->r_nlist = new_nlist;
}

/* Search link maps in all namespaces for the DSO that contains the object at
   address ADDR.  Returns the pointer to the link map of the matching DSO, or
   NULL if a match is not found.  */
struct link_map *
_dl_find_dso_for_object (const ElfW(Addr) addr)
{
  struct link_map *l;

  /* Find the highest-addressed object that ADDR is not below.  */
  for (Lmid_t ns = 0; ns < GL(dl_nns); ++ns)
    for (l = GL(dl_ns)[ns]._ns_loaded; l != NULL; l = l->l_next)
      if (addr >= l->l_map_start && addr < l->l_map_end
	  && (l->l_contiguous
	      || _dl_addr_inside_object (l, (ElfW(Addr)) addr)))
	{
	  assert (ns == l->l_ns);
	  return l;
	}
  return NULL;
}
rtld_hidden_def (_dl_find_dso_for_object);

/* Return true if NEW is found in the scope for MAP.  */
static size_t
scope_has_map (struct link_map *map, struct link_map *new)
{
  size_t cnt;
  for (cnt = 0; map->l_scope[cnt] != NULL; ++cnt)
    if (map->l_scope[cnt] == &new->l_searchlist)
      return true;
  return false;
}

/* Return the length of the scope for MAP.  */
static size_t
scope_size (struct link_map *map)
{
  size_t cnt;
  for (cnt = 0; map->l_scope[cnt] != NULL; )
    ++cnt;
  return cnt;
}

/* Resize the scopes of depended-upon objects, so that the new object
   can be added later without further allocation of memory.  This
   function can raise an exceptions due to malloc failure.  */
static void
resize_scopes (struct link_map *new)
{
  /* If the file is not loaded now as a dependency, add the search
     list of the newly loaded object to the scope.  */
  for (unsigned int i = 0; i < new->l_searchlist.r_nlist; ++i)
    {
      struct link_map *imap = new->l_searchlist.r_list[i];

      /* If the initializer has been called already, the object has
	 not been loaded here and now.  */
      if (imap->l_init_called && imap->l_type == lt_loaded)
	{
	  if (scope_has_map (imap, new))
	    /* Avoid duplicates.  */
	    continue;

	  size_t cnt = scope_size (imap);
	  if (__glibc_unlikely (cnt + 1 >= imap->l_scope_max))
	    {
	      /* The l_scope array is too small.  Allocate a new one
		 dynamically.  */
	      size_t new_size;
	      struct r_scope_elem **newp;

	      if (imap->l_scope != imap->l_scope_mem
		  && imap->l_scope_max < array_length (imap->l_scope_mem))
		{
		  /* If the current l_scope memory is not pointing to
		     the static memory in the structure, but the
		     static memory in the structure is large enough to
		     use for cnt + 1 scope entries, then switch to
		     using the static memory.  */
		  new_size = array_length (imap->l_scope_mem);
		  newp = imap->l_scope_mem;
		}
	      else
		{
		  new_size = imap->l_scope_max * 2;
		  newp = (struct r_scope_elem **)
		    malloc (new_size * sizeof (struct r_scope_elem *));
		  if (newp == NULL)
		    _dl_signal_error (ENOMEM, "dlopen", NULL,
				      N_("cannot create scope list"));
		}

	      /* Copy the array and the terminating NULL.  */
	      memcpy (newp, imap->l_scope,
		      (cnt + 1) * sizeof (imap->l_scope[0]));
	      struct r_scope_elem **old = imap->l_scope;

	      imap->l_scope = newp;

	      if (old != imap->l_scope_mem)
		_dl_scope_free (old);

	      imap->l_scope_max = new_size;
	    }
	}
    }
}

/* Second stage of resize_scopes: Add NEW to the scopes.  Also print
   debugging information about scopes if requested.

   This function cannot raise an exception because all required memory
   has been allocated by a previous call to resize_scopes.  */
static void
update_scopes (struct link_map *new)
{
  for (unsigned int i = 0; i < new->l_searchlist.r_nlist; ++i)
    {
      struct link_map *imap = new->l_searchlist.r_list[i];
      int from_scope = 0;

      if (imap->l_init_called && imap->l_type == lt_loaded)
	{
	  if (scope_has_map (imap, new))
	    /* Avoid duplicates.  */
	    continue;

	  size_t cnt = scope_size (imap);
	  /* Assert that resize_scopes has sufficiently enlarged the
	     array.  */
	  assert (cnt + 1 < imap->l_scope_max);

	  /* First terminate the extended list.  Otherwise a thread
	     might use the new last element and then use the garbage
	     at offset IDX+1.  */
	  imap->l_scope[cnt + 1] = NULL;
	  atomic_write_barrier ();
	  imap->l_scope[cnt] = &new->l_searchlist;

	  from_scope = cnt;
	}

      /* Print scope information.  */
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_SCOPES))
	_dl_show_scope (imap, from_scope);
    }
}

/* Call _dl_add_to_slotinfo with DO_ADD set to false, to allocate
   space in GL (dl_tls_dtv_slotinfo_list).  This can raise an
   exception.  The return value is true if any of the new objects use
   TLS.  */
static bool
resize_tls_slotinfo (struct link_map *new)
{
  bool any_tls = false;
  for (unsigned int i = 0; i < new->l_searchlist.r_nlist; ++i)
    {
      struct link_map *imap = new->l_searchlist.r_list[i];

      /* Only add TLS memory if this object is loaded now and
	 therefore is not yet initialized.  */
      if (! imap->l_init_called && imap->l_tls_blocksize > 0)
	{
	  _dl_add_to_slotinfo (imap, false);
	  any_tls = true;
	}
    }
  return any_tls;
}

/* Second stage of TLS update, after resize_tls_slotinfo.  This
   function does not raise any exception.  It should only be called if
   resize_tls_slotinfo returned true.  */
static void
update_tls_slotinfo (struct link_map *new)
{
  unsigned int first_static_tls = new->l_searchlist.r_nlist;
  for (unsigned int i = 0; i < new->l_searchlist.r_nlist; ++i)
    {
      struct link_map *imap = new->l_searchlist.r_list[i];

      /* Only add TLS memory if this object is loaded now and
	 therefore is not yet initialized.  */
      if (! imap->l_init_called && imap->l_tls_blocksize > 0)
	{
	  _dl_add_to_slotinfo (imap, true);

	  if (imap->l_need_tls_init
	      && first_static_tls == new->l_searchlist.r_nlist)
	    first_static_tls = i;
	}
    }

  size_t newgen = GL(dl_tls_generation) + 1;
  if (__glibc_unlikely (newgen == 0))
    _dl_fatal_printf (N_("\
TLS generation counter wrapped!  Please report this."));
  /* Can be read concurrently.  */
  atomic_store_relaxed (&GL(dl_tls_generation), newgen);

  /* We need a second pass for static tls data, because
     _dl_update_slotinfo must not be run while calls to
     _dl_add_to_slotinfo are still pending.  */
  for (unsigned int i = first_static_tls; i < new->l_searchlist.r_nlist; ++i)
    {
      struct link_map *imap = new->l_searchlist.r_list[i];

      if (imap->l_need_tls_init
	  && ! imap->l_init_called
	  && imap->l_tls_blocksize > 0)
	{
	  /* For static TLS we have to allocate the memory here and
	     now, but we can delay updating the DTV.  */
	  imap->l_need_tls_init = 0;
#ifdef SHARED
	  /* Update the slot information data for at least the
	     generation of the DSO we are allocating data for.  */

	  /* FIXME: This can terminate the process on memory
	     allocation failure.  It is not possible to raise
	     exceptions from this context; to fix this bug,
	     _dl_update_slotinfo would have to be split into two
	     operations, similar to resize_scopes and update_scopes
	     above.  This is related to bug 16134.  */
	  _dl_update_slotinfo (imap->l_tls_modid);
#endif

	  dl_init_static_tls (imap);
	  assert (imap->l_need_tls_init == 0);
	}
    }
}

/* Mark the objects as NODELETE if required.  This is delayed until
   after dlopen failure is not possible, so that _dl_close can clean
   up objects if necessary.  */
static void
activate_nodelete (struct link_map *new)
{
  /* It is necessary to traverse the entire namespace.  References to
     objects in the global scope and unique symbol bindings can force
     NODELETE status for objects outside the local scope.  */
  for (struct link_map *l = GL (dl_ns)[new->l_ns]._ns_loaded; l != NULL;
       l = l->l_next)
    if (l->l_nodelete_pending)
      {
	if (__glibc_unlikely (GLRO (dl_debug_mask) & DL_DEBUG_FILES))
	  _dl_debug_printf ("activating NODELETE for %s [%lu]\n",
			    l->l_name, l->l_ns);

	/* The flag can already be true at this point, e.g. a signal
	   handler may have triggered lazy binding and set NODELETE
	   status immediately.  */
	l->l_nodelete_active = true;

	/* This is just a debugging aid, to indicate that
	   activate_nodelete has run for this map.  */
	l->l_nodelete_pending = false;
      }
}

/* struct dl_init_args and call_dl_init are used to call _dl_init with
   exception handling disabled.  */
struct dl_init_args
{
  struct link_map *new;
  int argc;
  char **argv;
  char **env;
};

static void
call_dl_init (void *closure)
{
  struct dl_init_args *args = closure;
  _dl_init (args->new, args->argc, args->argv, args->env);
}

static void
dl_open_worker (void *a)
{
  struct dl_open_args *args = a;
  const char *file = args->file;
  int mode = args->mode;
  struct link_map *call_map = NULL;

  /* Determine the caller's map if necessary.  This is needed in case
     we have a DST, when we don't know the namespace ID we have to put
     the new object in, or when the file name has no path in which
     case we need to look along the RUNPATH/RPATH of the caller.  */
  const char *dst = strchr (file, '$');
  if (dst != NULL || args->nsid == __LM_ID_CALLER
      || strchr (file, '/') == NULL)
    {
      const void *caller_dlopen = args->caller_dlopen;

      /* We have to find out from which object the caller is calling.
	 By default we assume this is the main application.  */
      call_map = GL(dl_ns)[LM_ID_BASE]._ns_loaded;

      struct link_map *l = _dl_find_dso_for_object ((ElfW(Addr)) caller_dlopen);

      if (l)
	call_map = l;

      if (args->nsid == __LM_ID_CALLER)
	args->nsid = call_map->l_ns;
    }

  /* The namespace ID is now known.  Keep track of whether libc.so was
     already loaded, to determine whether it is necessary to call the
     early initialization routine (or clear libc_map on error).  */
  args->libc_already_loaded = GL(dl_ns)[args->nsid].libc_map != NULL;

  /* Retain the old value, so that it can be restored.  */
  args->original_global_scope_pending_adds
    = GL (dl_ns)[args->nsid]._ns_global_scope_pending_adds;

  /* One might be tempted to assert that we are RT_CONSISTENT at this point, but that
     may not be true if this is a recursive call to dlopen.  */
  _dl_debug_initialize (0, args->nsid);

  /* Load the named object.  */
  struct link_map *new;
  args->map = new = _dl_map_object (call_map, file, lt_loaded, 0,
				    mode | __RTLD_CALLMAP, args->nsid);

  /* If the pointer returned is NULL this means the RTLD_NOLOAD flag is
     set and the object is not already loaded.  */
  if (new == NULL)
    {
      assert (mode & RTLD_NOLOAD);
      return;
    }

  if (__glibc_unlikely (mode & __RTLD_SPROF))
    /* This happens only if we load a DSO for 'sprof'.  */
    return;

  /* This object is directly loaded.  */
  ++new->l_direct_opencount;

  /* It was already open.  */
  if (__glibc_unlikely (new->l_searchlist.r_list != NULL))
    {
      /* Let the user know about the opencount.  */
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_FILES))
	_dl_debug_printf ("opening file=%s [%lu]; direct_opencount=%u\n\n",
			  new->l_name, new->l_ns, new->l_direct_opencount);

      /* If the user requested the object to be in the global
	 namespace but it is not so far, prepare to add it now.  This
	 can raise an exception to do a malloc failure.  */
      if ((mode & RTLD_GLOBAL) && new->l_global == 0)
	add_to_global_resize (new);

      /* Mark the object as not deletable if the RTLD_NODELETE flags
	 was passed.  */
      if (__glibc_unlikely (mode & RTLD_NODELETE))
	{
	  if (__glibc_unlikely (GLRO (dl_debug_mask) & DL_DEBUG_FILES)
	      && !new->l_nodelete_active)
	    _dl_debug_printf ("marking %s [%lu] as NODELETE\n",
			      new->l_name, new->l_ns);
	  new->l_nodelete_active = true;
	}

      /* Finalize the addition to the global scope.  */
      if ((mode & RTLD_GLOBAL) && new->l_global == 0)
	add_to_global_update (new);

      assert (_dl_debug_initialize (0, args->nsid)->r_state == RT_CONSISTENT);

      return;
    }

  /* Schedule NODELETE marking for the directly loaded object if
     requested.  */
  if (__glibc_unlikely (mode & RTLD_NODELETE))
    new->l_nodelete_pending = true;

  /* Load that object's dependencies.  */
  _dl_map_object_deps (new, NULL, 0, 0,
		       mode & (__RTLD_DLOPEN | RTLD_DEEPBIND | __RTLD_AUDIT));

  /* So far, so good.  Now check the versions.  */
  for (unsigned int i = 0; i < new->l_searchlist.r_nlist; ++i)
    if (new->l_searchlist.r_list[i]->l_real->l_versions == NULL)
      {
	struct link_map *map = new->l_searchlist.r_list[i]->l_real;
	_dl_check_map_versions (map, 0, 0);
#ifndef SHARED
	/* During static dlopen, check if ld.so has been loaded.
	   Perform partial initialization in this case.  This must
	   come after the symbol versioning initialization in
	   _dl_check_map_versions.  */
	if (map->l_info[DT_SONAME] != NULL
	    && strcmp (((const char *) D_PTR (map, l_info[DT_STRTAB])
			+ map->l_info[DT_SONAME]->d_un.d_val), LD_SO) == 0)
	  __rtld_static_init (map);
#endif
      }

#ifdef SHARED
  /* Auditing checkpoint: we have added all objects.  */
  if (__glibc_unlikely (GLRO(dl_naudit) > 0))
    {
      struct link_map *head = GL(dl_ns)[new->l_ns]._ns_loaded;
      /* Do not call the functions for any auditing object.  */
      if (head->l_auditing == 0)
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

  /* Notify the debugger all new objects are now ready to go.  */
  struct r_debug *r = _dl_debug_initialize (0, args->nsid);
  r->r_state = RT_CONSISTENT;
  _dl_debug_state ();
  LIBC_PROBE (map_complete, 3, args->nsid, r, new);

  _dl_open_check (new);

  /* Print scope information.  */
  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_SCOPES))
    _dl_show_scope (new, 0);

  /* Only do lazy relocation if `LD_BIND_NOW' is not set.  */
  int reloc_mode = mode & __RTLD_AUDIT;
  if (GLRO(dl_lazy))
    reloc_mode |= mode & RTLD_LAZY;

  /* Objects must be sorted by dependency for the relocation process.
     This allows IFUNC relocations to work and it also means copy
     relocation of dependencies are if necessary overwritten.
     __dl_map_object_deps has already sorted l_initfini for us.  */
  unsigned int first = UINT_MAX;
  unsigned int last = 0;
  unsigned int j = 0;
  struct link_map *l = new->l_initfini[0];
  do
    {
      if (! l->l_real->l_relocated)
	{
	  if (first == UINT_MAX)
	    first = j;
	  last = j + 1;
	}
      l = new->l_initfini[++j];
    }
  while (l != NULL);

  int relocation_in_progress = 0;

  /* Perform relocation.  This can trigger lazy binding in IFUNC
     resolvers.  For NODELETE mappings, these dependencies are not
     recorded because the flag has not been applied to the newly
     loaded objects.  This means that upon dlopen failure, these
     NODELETE objects can be unloaded despite existing references to
     them.  However, such relocation dependencies in IFUNC resolvers
     are undefined anyway, so this is not a problem.  */

  for (unsigned int i = last; i-- > first; )
    {
      l = new->l_initfini[i];

      if (l->l_real->l_relocated)
	continue;

      if (! relocation_in_progress)
	{
	  /* Notify the debugger that relocations are about to happen.  */
	  LIBC_PROBE (reloc_start, 2, args->nsid, r);
	  relocation_in_progress = 1;
	}

#ifdef SHARED
      if (__glibc_unlikely (GLRO(dl_profile) != NULL))
	{
	  /* If this here is the shared object which we want to profile
	     make sure the profile is started.  We can find out whether
	     this is necessary or not by observing the `_dl_profile_map'
	     variable.  If it was NULL but is not NULL afterwards we must
	     start the profiling.  */
	  struct link_map *old_profile_map = GL(dl_profile_map);

	  _dl_relocate_object (l, l->l_scope, reloc_mode | RTLD_LAZY, 1);

	  if (old_profile_map == NULL && GL(dl_profile_map) != NULL)
	    {
	      /* We must prepare the profiling.  */
	      _dl_start_profile ();

	      /* Prevent unloading the object.  */
	      GL(dl_profile_map)->l_nodelete_active = true;
	    }
	}
      else
#endif
	_dl_relocate_object (l, l->l_scope, reloc_mode, 0);
    }

  /* This only performs the memory allocations.  The actual update of
     the scopes happens below, after failure is impossible.  */
  resize_scopes (new);

  /* Increase the size of the GL (dl_tls_dtv_slotinfo_list) data
     structure.  */
  bool any_tls = resize_tls_slotinfo (new);

  /* Perform the necessary allocations for adding new global objects
     to the global scope below.  */
  if (mode & RTLD_GLOBAL)
    add_to_global_resize (new);

  /* Demarcation point: After this, no recoverable errors are allowed.
     All memory allocations for new objects must have happened
     before.  */

  /* Finalize the NODELETE status first.  This comes before
     update_scopes, so that lazy binding will not see pending NODELETE
     state for newly loaded objects.  There is a compiler barrier in
     update_scopes which ensures that the changes from
     activate_nodelete are visible before new objects show up in the
     local scope.  */
  activate_nodelete (new);

  /* Second stage after resize_scopes: Actually perform the scope
     update.  After this, dlsym and lazy binding can bind to new
     objects.  */
  update_scopes (new);

  /* FIXME: It is unclear whether the order here is correct.
     Shouldn't new objects be made available for binding (and thus
     execution) only after there TLS data has been set up fully?
     Fixing bug 16134 will likely make this distinction less
     important.  */

  /* Second stage after resize_tls_slotinfo: Update the slotinfo data
     structures.  */
  if (any_tls)
    /* FIXME: This calls _dl_update_slotinfo, which aborts the process
       on memory allocation failure.  See bug 16134.  */
    update_tls_slotinfo (new);

  /* Notify the debugger all new objects have been relocated.  */
  if (relocation_in_progress)
    LIBC_PROBE (reloc_complete, 3, args->nsid, r, new);

  /* If libc.so was not there before, attempt to call its early
     initialization routine.  Indicate to the initialization routine
     whether the libc being initialized is the one in the base
     namespace.  */
  if (!args->libc_already_loaded)
    {
      /* dlopen cannot be used to load an initial libc by design.  */
      struct link_map *libc_map = GL(dl_ns)[args->nsid].libc_map;
      _dl_call_libc_early_init (libc_map, false);
    }

  /* Run the initializer functions of new objects.  Temporarily
     disable the exception handler, so that lazy binding failures are
     fatal.  */
  {
    struct dl_init_args init_args =
      {
        .new = new,
        .argc = args->argc,
        .argv = args->argv,
        .env = args->env
      };
    _dl_catch_exception (NULL, call_dl_init, &init_args);
  }

  /* Now we can make the new map available in the global scope.  */
  if (mode & RTLD_GLOBAL)
    add_to_global_update (new);

  /* Let the user know about the opencount.  */
  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_FILES))
    _dl_debug_printf ("opening file=%s [%lu]; direct_opencount=%u\n\n",
		      new->l_name, new->l_ns, new->l_direct_opencount);
}

void *
_dl_open (const char *file, int mode, const void *caller_dlopen, Lmid_t nsid,
	  int argc, char *argv[], char *env[])
{
  if ((mode & RTLD_BINDING_MASK) == 0)
    /* One of the flags must be set.  */
    _dl_signal_error (EINVAL, file, NULL, N_("invalid mode for dlopen()"));

  /* Make sure we are alone.  */
  __rtld_lock_lock_recursive (GL(dl_load_lock));

  if (__glibc_unlikely (nsid == LM_ID_NEWLM))
    {
      /* Find a new namespace.  */
      for (nsid = 1; DL_NNS > 1 && nsid < GL(dl_nns); ++nsid)
	if (GL(dl_ns)[nsid]._ns_loaded == NULL)
	  break;

      if (__glibc_unlikely (nsid == DL_NNS))
	{
	  /* No more namespace available.  */
	  __rtld_lock_unlock_recursive (GL(dl_load_lock));

	  _dl_signal_error (EINVAL, file, NULL, N_("\
no more namespaces available for dlmopen()"));
	}
      else if (nsid == GL(dl_nns))
	{
	  __rtld_lock_initialize (GL(dl_ns)[nsid]._ns_unique_sym_table.lock);
	  ++GL(dl_nns);
	}

      _dl_debug_initialize (0, nsid)->r_state = RT_CONSISTENT;
    }
  /* Never allow loading a DSO in a namespace which is empty.  Such
     direct placements is only causing problems.  Also don't allow
     loading into a namespace used for auditing.  */
  else if (__glibc_unlikely (nsid != LM_ID_BASE && nsid != __LM_ID_CALLER)
	   && (__glibc_unlikely (nsid < 0 || nsid >= GL(dl_nns))
	       /* This prevents the [NSID] index expressions from being
		  evaluated, so the compiler won't think that we are
		  accessing an invalid index here in the !SHARED case where
		  DL_NNS is 1 and so any NSID != 0 is invalid.  */
	       || DL_NNS == 1
	       || GL(dl_ns)[nsid]._ns_nloaded == 0
	       || GL(dl_ns)[nsid]._ns_loaded->l_auditing))
    _dl_signal_error (EINVAL, file, NULL,
		      N_("invalid target namespace in dlmopen()"));

  struct dl_open_args args;
  args.file = file;
  args.mode = mode;
  args.caller_dlopen = caller_dlopen;
  args.map = NULL;
  args.nsid = nsid;
  /* args.libc_already_loaded is always assigned by dl_open_worker
     (before any explicit/non-local returns).  */
  args.argc = argc;
  args.argv = argv;
  args.env = env;

  struct dl_exception exception;
  int errcode = _dl_catch_exception (&exception, dl_open_worker, &args);

#if defined USE_LDCONFIG && !defined MAP_COPY
  /* We must unmap the cache file.  */
  _dl_unload_cache ();
#endif

  /* Do this for both the error and success cases.  The old value has
     only been determined if the namespace ID was assigned (i.e., it
     is not __LM_ID_CALLER).  In the success case, we actually may
     have consumed more pending adds than planned (because the local
     scopes overlap in case of a recursive dlopen, the inner dlopen
     doing some of the globalization work of the outer dlopen), so the
     old pending adds value is larger than absolutely necessary.
     Since it is just a conservative upper bound, this is harmless.
     The top-level dlopen call will restore the field to zero.  */
  if (args.nsid >= 0)
    GL (dl_ns)[args.nsid]._ns_global_scope_pending_adds
      = args.original_global_scope_pending_adds;

  /* See if an error occurred during loading.  */
  if (__glibc_unlikely (exception.errstring != NULL))
    {
      /* Avoid keeping around a dangling reference to the libc.so link
	 map in case it has been cached in libc_map.  */
      if (!args.libc_already_loaded)
	GL(dl_ns)[nsid].libc_map = NULL;

      /* Remove the object from memory.  It may be in an inconsistent
	 state if relocation failed, for example.  */
      if (args.map)
	{
	  _dl_close_worker (args.map, true);

	  /* All l_nodelete_pending objects should have been deleted
	     at this point, which is why it is not necessary to reset
	     the flag here.  */
	}

      assert (_dl_debug_initialize (0, args.nsid)->r_state == RT_CONSISTENT);

      /* Release the lock.  */
      __rtld_lock_unlock_recursive (GL(dl_load_lock));

      /* Reraise the error.  */
      _dl_signal_exception (errcode, &exception, NULL);
    }

  assert (_dl_debug_initialize (0, args.nsid)->r_state == RT_CONSISTENT);

  /* Release the lock.  */
  __rtld_lock_unlock_recursive (GL(dl_load_lock));

  return args.map;
}


void
_dl_show_scope (struct link_map *l, int from)
{
  _dl_debug_printf ("object=%s [%lu]\n",
		    DSO_FILENAME (l->l_name), l->l_ns);
  if (l->l_scope != NULL)
    for (int scope_cnt = from; l->l_scope[scope_cnt] != NULL; ++scope_cnt)
      {
	_dl_debug_printf (" scope %u:", scope_cnt);

	for (unsigned int cnt = 0; cnt < l->l_scope[scope_cnt]->r_nlist; ++cnt)
	  if (*l->l_scope[scope_cnt]->r_list[cnt]->l_name)
	    _dl_debug_printf_c (" %s",
				l->l_scope[scope_cnt]->r_list[cnt]->l_name);
	  else
	    _dl_debug_printf_c (" %s", RTLD_PROGNAME);

	_dl_debug_printf_c ("\n");
      }
  else
    _dl_debug_printf (" no scope\n");
  _dl_debug_printf ("\n");
}
