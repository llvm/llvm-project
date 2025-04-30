/* Handle loading and unloading shared objects for internal libc purposes.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.columbia.edu>, 1999.

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

#include <dlfcn.h>
#include <stdlib.h>
#include <ldsodefs.h>
#include <dl-hash.h>

extern int __libc_argc attribute_hidden;
extern char **__libc_argv attribute_hidden;

extern char **__environ;

/* The purpose of this file is to provide wrappers around the dynamic
   linker error mechanism (similar to dlopen() et al in libdl) which
   are usable from within libc.  Generally we want to throw away the
   string that dlerror() would return and just pass back a null pointer
   for errors.  This also lets the rest of libc not know about the error
   handling mechanism.

   Much of this code came from gconv_dl.c with slight modifications. */

static int
dlerror_run (void (*operate) (void *), void *args)
{
  const char *objname;
  const char *last_errstring = NULL;
  bool malloced;

  int result = (GLRO (dl_catch_error) (&objname, &last_errstring, &malloced,
				       operate, args)
		?: last_errstring != NULL);

  if (result && malloced)
    GLRO (dl_error_free) ((char *) last_errstring);

  return result;
}

/* These functions are called by dlerror_run... */

struct do_dlopen_args
{
  /* Argument to do_dlopen.  */
  const char *name;
  /* Opening mode.  */
  int mode;
  /* This is the caller of the dlopen() function.  */
  const void *caller_dlopen;

  /* Return from do_dlopen.  */
  struct link_map *map;
};

struct do_dlsym_args
{
  /* Arguments to do_dlsym.  */
  struct link_map *map;
  const char *name;

  /* Return values of do_dlsym.  */
  lookup_t loadbase;
  const ElfW(Sym) *ref;
};

struct do_dlvsym_args
{
  /* dlvsym is like dlsym.  */
  struct do_dlsym_args dlsym;

  /* But dlvsym needs a version  as well.  */
  struct r_found_version version;
};

static void
do_dlopen (void *ptr)
{
  struct do_dlopen_args *args = (struct do_dlopen_args *) ptr;
  /* Open and relocate the shared object.  */
  args->map = GLRO(dl_open) (args->name, args->mode, args->caller_dlopen,
			     __LM_ID_CALLER, __libc_argc, __libc_argv,
			     __environ);
}

static void
do_dlsym (void *ptr)
{
  struct do_dlsym_args *args = (struct do_dlsym_args *) ptr;
  args->ref = NULL;
  args->loadbase = GLRO(dl_lookup_symbol_x) (args->name, args->map, &args->ref,
					     args->map->l_local_scope, NULL, 0,
					     DL_LOOKUP_RETURN_NEWEST, NULL);
}

static void
do_dlvsym (void *ptr)
{
  struct do_dlvsym_args *args = ptr;
  args->dlsym.ref = NULL;
  args->dlsym.loadbase
    = GLRO(dl_lookup_symbol_x) (args->dlsym.name, args->dlsym.map,
				&args->dlsym.ref,
				args->dlsym.map->l_local_scope,
				&args->version, 0, 0, NULL);
}

static void
do_dlclose (void *ptr)
{
  GLRO(dl_close) ((struct link_map *) ptr);
}

#ifndef SHARED
static void
do_dlsym_private (void *ptr)
{
  lookup_t l;
  struct r_found_version vers;
  vers.name = "GLIBC_PRIVATE";
  vers.hidden = 1;
  /* vers.hash = _dl_elf_hash (vers.name);  */
  vers.hash = 0x0963cf85;
  vers.filename = NULL;

  struct do_dlsym_args *args = (struct do_dlsym_args *) ptr;
  args->ref = NULL;
  l = GLRO(dl_lookup_symbol_x) (args->name, args->map, &args->ref,
				args->map->l_scope, &vers, 0, 0, NULL);
  args->loadbase = l;
}
#endif

/* ... and these functions call dlerror_run. */

void *
__libc_dlopen_mode (const char *name, int mode)
{
  struct do_dlopen_args args;
  args.name = name;
  args.mode = mode;
  args.caller_dlopen = RETURN_ADDRESS (0);

#ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->libc_dlopen_mode (name, mode);
#endif
  return dlerror_run (do_dlopen, &args) ? NULL : (void *) args.map;
}

#ifndef SHARED
void *
__libc_dlsym_private (struct link_map *map, const char *name)
{
  struct do_dlsym_args sargs;
  sargs.map = map;
  sargs.name = name;

  if (! dlerror_run (do_dlsym_private, &sargs))
    return DL_SYMBOL_ADDRESS (sargs.loadbase, sargs.ref);
  return NULL;
}
#endif

void *
__libc_dlsym (void *map, const char *name)
{
  struct do_dlsym_args args;
  args.map = map;
  args.name = name;

#ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->libc_dlsym (map, name);
#endif
  return (dlerror_run (do_dlsym, &args) ? NULL
	  : (void *) (DL_SYMBOL_ADDRESS (args.loadbase, args.ref)));
}

/* Replacement for dlvsym.  MAP must be a real map.  This function
   returns NULL without setting the dlerror value in case of static
   dlopen from an old binary.  */
void *
__libc_dlvsym (void *map, const char *name, const char *version)
{
#ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->libc_dlvsym (map, name, version);
#endif

  struct do_dlvsym_args args;
  args.dlsym.map = map;
  args.dlsym.name = name;

  /* See _dl_vsym in dl-sym.c.  */
  args.version.name = version;
  args.version.hidden = 1;
  args.version.hash = _dl_elf_hash (version);
  args.version.filename = NULL;

  return (dlerror_run (do_dlvsym, &args) ? NULL
	  : (void *) (DL_SYMBOL_ADDRESS (args.dlsym.loadbase,
					 args.dlsym.ref)));
}

int
__libc_dlclose (void *map)
{
#ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->libc_dlclose (map);
#endif
  return dlerror_run (do_dlclose, map);
}


static bool __libc_freeres_fn_section
free_slotinfo (struct dtv_slotinfo_list **elemp)
{
  size_t cnt;

  if (*elemp == NULL)
    /* Nothing here, all is removed (or there never was anything).  */
    return true;

  if (!free_slotinfo (&(*elemp)->next))
    /* We cannot free the entry.  */
    return false;

  /* That cleared our next pointer for us.  */

  for (cnt = 0; cnt < (*elemp)->len; ++cnt)
    if ((*elemp)->slotinfo[cnt].map != NULL)
      /* Still used.  */
      return false;

  /* We can remove the list element.  */
  free (*elemp);
  *elemp = NULL;

  return true;
}


libc_freeres_fn (free_mem)
{
  struct link_map *l;
  struct r_search_path_elem *d;

  /* Remove all search directories.  */
  d = GL(dl_all_dirs);
  while (d != GLRO(dl_init_all_dirs))
    {
      struct r_search_path_elem *old = d;
      d = d->next;
      free (old);
    }

  for (Lmid_t ns = 0; ns < GL(dl_nns); ++ns)
    {
      for (l = GL(dl_ns)[ns]._ns_loaded; l != NULL; l = l->l_next)
	{
	  struct libname_list *lnp = l->l_libname->next;

	  l->l_libname->next = NULL;

	  /* Remove all additional names added to the objects.  */
	  while (lnp != NULL)
	    {
	      struct libname_list *old = lnp;
	      lnp = lnp->next;
	      if (! old->dont_free)
		free (old);
	    }

	  /* Free the initfini dependency list.  */
	  if (l->l_free_initfini)
	    free (l->l_initfini);
	  l->l_initfini = NULL;
	}

      if (__builtin_expect (GL(dl_ns)[ns]._ns_global_scope_alloc, 0) != 0
	  && (GL(dl_ns)[ns]._ns_main_searchlist->r_nlist
	      // XXX Check whether we need NS-specific initial_searchlist
	      == GLRO(dl_initial_searchlist).r_nlist))
	{
	  /* All object dynamically loaded by the program are unloaded.  Free
	     the memory allocated for the global scope variable.  */
	  struct link_map **old = GL(dl_ns)[ns]._ns_main_searchlist->r_list;

	  /* Put the old map in.  */
	  GL(dl_ns)[ns]._ns_main_searchlist->r_list
	    // XXX Check whether we need NS-specific initial_searchlist
	    = GLRO(dl_initial_searchlist).r_list;
	  /* Signal that the original map is used.  */
	  GL(dl_ns)[ns]._ns_global_scope_alloc = 0;

	  /* Now free the old map.  */
	  free (old);
	}
    }

  /* Free the memory allocated for the dtv slotinfo array.  We can do
     this only if all modules which used this memory are unloaded.  */
#ifdef SHARED
  if (GL(dl_initial_dtv) == NULL)
    /* There was no initial TLS setup, it was set up later when
       it used the normal malloc.  */
    free_slotinfo (&GL(dl_tls_dtv_slotinfo_list));
  else
#endif
    /* The first element of the list does not have to be deallocated.
       It was allocated in the dynamic linker (i.e., with a different
       malloc), and in the static library it's in .bss space.  */
    free_slotinfo (&GL(dl_tls_dtv_slotinfo_list)->next);

  void *scope_free_list = GL(dl_scope_free_list);
  GL(dl_scope_free_list) = NULL;
  free (scope_free_list);
}
