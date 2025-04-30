/* Storage management for the chain of loaded shared objects.
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

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <ldsodefs.h>

#include <assert.h>


/* Add the new link_map NEW to the end of the namespace list.  */
void
_dl_add_to_namespace_list (struct link_map *new, Lmid_t nsid)
{
  /* We modify the list of loaded objects.  */
  __rtld_lock_lock_recursive (GL(dl_load_write_lock));

  if (GL(dl_ns)[nsid]._ns_loaded != NULL)
    {
      struct link_map *l = GL(dl_ns)[nsid]._ns_loaded;
      while (l->l_next != NULL)
	l = l->l_next;
      new->l_prev = l;
      /* new->l_next = NULL;   Would be necessary but we use calloc.  */
      l->l_next = new;
    }
  else
    GL(dl_ns)[nsid]._ns_loaded = new;
  ++GL(dl_ns)[nsid]._ns_nloaded;
  new->l_serial = GL(dl_load_adds);
  ++GL(dl_load_adds);

  __rtld_lock_unlock_recursive (GL(dl_load_write_lock));
}


/* Allocate a `struct link_map' for a new object being loaded,
   and enter it into the _dl_loaded list.  */
struct link_map *
_dl_new_object (char *realname, const char *libname, int type,
		struct link_map *loader, int mode, Lmid_t nsid)
{
#ifdef SHARED
  unsigned int naudit;
  if (__glibc_unlikely ((mode & __RTLD_OPENEXEC) != 0))
    {
      assert (type == lt_executable);
      assert (nsid == LM_ID_BASE);

      /* Ignore the specified libname for the main executable.  It is
	 only known with an explicit loader invocation.  */
      libname = "";

      /* We create the map for the executable before we know whether
	 we have auditing libraries and if yes, how many.  Assume the
	 worst.  */
      naudit = DL_NNS;
    }
  else
    naudit = GLRO (dl_naudit);
#endif

  size_t libname_len = strlen (libname) + 1;
  struct link_map *new;
  struct libname_list *newname;
#ifdef SHARED
  size_t audit_space = naudit * sizeof (struct auditstate);
#else
# define audit_space 0
#endif

  new = (struct link_map *) calloc (sizeof (*new) + audit_space
				    + sizeof (struct link_map *)
				    + sizeof (*newname) + libname_len, 1);
  if (new == NULL)
    return NULL;

  new->l_real = new;
  new->l_symbolic_searchlist.r_list = (struct link_map **) ((char *) (new + 1)
							    + audit_space);

  new->l_libname = newname
    = (struct libname_list *) (new->l_symbolic_searchlist.r_list + 1);
  newname->name = (char *) memcpy (newname + 1, libname, libname_len);
  /* newname->next = NULL;	We use calloc therefore not necessary.  */
  newname->dont_free = 1;

  /* When we create the executable link map, or a VDSO link map, we start
     with "" for the l_name. In these cases "" points to ld.so rodata
     and won't get dumped during core file generation. Therefore to assist
     gdb and to create more self-contained core files we adjust l_name to
     point at the newly allocated copy (which will get dumped) instead of
     the ld.so rodata copy.

     Furthermore, in case of explicit loader invocation, discard the
     name of the main executable, to match the regular behavior, where
     name of the executable is not known.  */
#ifdef SHARED
  if (*realname != '\0' && (mode & __RTLD_OPENEXEC) == 0)
#else
  if (*realname != '\0')
#endif
    new->l_name = realname;
  else
    new->l_name = (char *) newname->name + libname_len - 1;

  new->l_type = type;
  /* If we set the bit now since we know it is never used we avoid
     dirtying the cache line later.  */
  if ((GLRO(dl_debug_mask) & DL_DEBUG_UNUSED) == 0)
    new->l_used = 1;
  new->l_loader = loader;
#if NO_TLS_OFFSET != 0
  new->l_tls_offset = NO_TLS_OFFSET;
#endif
  new->l_ns = nsid;

#ifdef SHARED
  for (unsigned int cnt = 0; cnt < naudit; ++cnt)
    /* No need to initialize bindflags due to calloc.  */
    link_map_audit_state (new, cnt)->cookie = (uintptr_t) new;
#endif

  /* new->l_global = 0;	We use calloc therefore not necessary.  */

  /* Use the 'l_scope_mem' array by default for the 'l_scope'
     information.  If we need more entries we will allocate a large
     array dynamically.  */
  new->l_scope = new->l_scope_mem;
  new->l_scope_max = sizeof (new->l_scope_mem) / sizeof (new->l_scope_mem[0]);

  /* Counter for the scopes we have to handle.  */
  int idx = 0;

  if (GL(dl_ns)[nsid]._ns_loaded != NULL)
    /* Add the global scope.  */
    new->l_scope[idx++] = &GL(dl_ns)[nsid]._ns_loaded->l_searchlist;

  /* If we have no loader the new object acts as it.  */
  if (loader == NULL)
    loader = new;
  else
    /* Determine the local scope.  */
    while (loader->l_loader != NULL)
      loader = loader->l_loader;

  /* Insert the scope if it isn't the global scope we already added.  */
  if (idx == 0 || &loader->l_searchlist != new->l_scope[0])
    {
      if ((mode & RTLD_DEEPBIND) != 0 && idx != 0)
	{
	  new->l_scope[1] = new->l_scope[0];
	  idx = 0;
	}

      new->l_scope[idx] = &loader->l_searchlist;
    }

  new->l_local_scope[0] = &new->l_searchlist;

  /* Determine the origin.  If allocating the link map for the main
     executable, the realname is not known and "".  In this case, the
     origin needs to be determined by other means.  However, in case
     of an explicit loader invocation, the pathname of the main
     executable is known and needs to be processed here: From the
     point of view of the kernel, the main executable is the
     dynamic loader, and this would lead to a computation of the wrong
     origin.  */
  if (realname[0] != '\0')
    {
      size_t realname_len = strlen (realname) + 1;
      char *origin;
      char *cp;

      if (realname[0] == '/')
	{
	  /* It is an absolute path.  Use it.  But we have to make a
	     copy since we strip out the trailing slash.  */
	  cp = origin = (char *) malloc (realname_len);
	  if (origin == NULL)
	    {
	      origin = (char *) -1;
	      goto out;
	    }
	}
      else
	{
	  size_t len = realname_len;
	  char *result = NULL;

	  /* Get the current directory name.  */
	  origin = NULL;
	  do
	    {
	      char *new_origin;

	      len += 128;
	      new_origin = (char *) realloc (origin, len);
	      if (new_origin == NULL)
		/* We exit the loop.  Note that result == NULL.  */
		break;
	      origin = new_origin;
	    }
	  while ((result = __getcwd (origin, len - realname_len)) == NULL
		 && errno == ERANGE);

	  if (result == NULL)
	    {
	      /* We were not able to determine the current directory.
		 Note that free(origin) is OK if origin == NULL.  */
	      free (origin);
	      origin = (char *) -1;
	      goto out;
	    }

	  /* Find the end of the path and see whether we have to add a
	     slash.  We could use rawmemchr but this need not be
	     fast.  */
	  cp = (strchr) (origin, '\0');
	  if (cp[-1] != '/')
	    *cp++ = '/';
	}

      /* Add the real file name.  */
      cp = __mempcpy (cp, realname, realname_len);

      /* Now remove the filename and the slash.  Leave the slash if
	 the name is something like "/foo".  */
      do
	--cp;
      while (*cp != '/');

      if (cp == origin)
	/* Keep the only slash which is the first character.  */
	++cp;
      *cp = '\0';

    out:
      new->l_origin = origin;
    }

  return new;
}
