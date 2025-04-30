/* Global list of NSS service modules.
   Copyright (c) 2020-2021 Free Software Foundation, Inc.
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

#include <nsswitch.h>
#include <nscd/nscd.h>
#include <nscd/nscd_proto.h>

#include <array_length.h>
#include <assert.h>
#include <atomic.h>
#include <dlfcn.h>
#include <gnu/lib-names.h>
#include <libc-lock.h>
#include <nss_dns.h>
#include <nss_files.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysdep.h>

/* Suffix after .so of NSS service modules.  This is a bit of magic,
   but we assume LIBNSS_FILES_SO looks like "libnss_files.so.2" and we
   want a pointer to the ".2" part.  We have no API to extract this
   except through the auto-generated lib-names.h and some static
   pointer manipulation.  The "-1" accounts for the trailing NUL
   included in the sizeof.  */
static const char *const __nss_shlib_revision
	= LIBNSS_FILES_SO + sizeof("libnss_files.so") - 1;

/* A single-linked list used to implement a mapping from service names
   to NSS modules.  (Most systems only use five or so modules, so a
   list is sufficient here.)  Elements of this list are never freed
   during normal operation.  */
static struct nss_module *nss_module_list;

/* Covers the list and also loading of individual NSS service
   modules.  */
__libc_lock_define (static, nss_module_list_lock);

#if defined USE_NSCD && (!defined DO_STATIC_NSS || defined SHARED)
/* Nonzero if this is the nscd process.  */
static bool is_nscd;
/* The callback passed to the init functions when nscd is used.  */
static void (*nscd_init_cb) (size_t, struct traced_file *);
#endif

/* Allocate the service NAME with length NAME_LENGTH.  If the service
   is already allocated in the nss_module_list cache then we return a
   pointer to the struct nss_module, otherwise we try to allocate a
   new struct nss_module entry and add it to the global
   nss_modules_list cache.  If we fail to allocate the entry we return
   NULL.  Failure to allocate the entry is always transient.  */
struct nss_module *
__nss_module_allocate (const char *name, size_t name_length)
{
  __libc_lock_lock (nss_module_list_lock);

  struct nss_module *result = NULL;
  for (struct nss_module *p = nss_module_list; p != NULL; p = p->next)
    if (strncmp (p->name, name, name_length) == 0
        && p->name[name_length] == '\0')
      {
        /* Return the previously existing object.  */
        result = p;
        break;
      }

  if (result == NULL)
    {
      /* Allocate a new list entry if the name was not found in the
         list.  */
      result = malloc (sizeof (*result) + name_length + 1);
      if (result != NULL)
        {
          result->state = nss_module_uninitialized;
          memcpy (result->name, name, name_length);
          result->name[name_length] = '\0';
          result->handle = NULL;
          result->next = nss_module_list;
          nss_module_list = result;
        }
    }

  __libc_lock_unlock (nss_module_list_lock);
  return result;
}

/* Long enough to store the name of any function in the
   nss_function_name_array list below, as getprotobynumber_r is the
   longest entry in that list.  */
typedef char function_name[sizeof("getprotobynumber_r")];

static const function_name nss_function_name_array[] =
  {
#undef DEFINE_NSS_FUNCTION
#define DEFINE_NSS_FUNCTION(x) #x,
#include "function.def"
  };

/* Loads a built-in module, binding the symbols using the supplied
   callback function.  Always returns true.  */
static bool
module_load_builtin (struct nss_module *module,
		     void (*bind) (nss_module_functions_untyped))
{
  /* Initialize the function pointers, following the double-checked
     locking idiom.  */
  __libc_lock_lock (nss_module_list_lock);
  switch ((enum nss_module_state) atomic_load_acquire (&module->state))
    {
    case nss_module_uninitialized:
    case nss_module_failed:
      bind (module->functions.untyped);

#ifdef PTR_MANGLE
      for (int i = 0; i < nss_module_functions_count; ++i)
	PTR_MANGLE (module->functions.untyped[i]);
#endif

      module->handle = NULL;
      /* Synchronizes with unlocked __nss_module_load atomic_load_acquire.  */
      atomic_store_release (&module->state, nss_module_loaded);
      break;
    case nss_module_loaded:
      /* Nothing to clean up.  */
      break;
    }
  __libc_lock_unlock (nss_module_list_lock);
  return true;
}

/* Loads the built-in nss_files module.  */
static bool
module_load_nss_files (struct nss_module *module)
{
#ifdef USE_NSCD
  if (is_nscd)
    {
      void (*cb) (size_t, struct traced_file *) = nscd_init_cb;
# ifdef PTR_DEMANGLE
      PTR_DEMANGLE (cb);
# endif
      _nss_files_init (cb);
    }
#endif
  return module_load_builtin (module, __nss_files_functions);
}

/* Loads the built-in nss_dns module.  */
static bool
module_load_nss_dns (struct nss_module *module)
{
  return module_load_builtin (module, __nss_dns_functions);
}

/* Internal implementation of __nss_module_load.  */
static bool
module_load (struct nss_module *module)
{
  if (strcmp (module->name, "files") == 0)
    return module_load_nss_files (module);
  if (strcmp (module->name, "dns") == 0)
    return module_load_nss_dns (module);

  void *handle;
  {
    char *shlib_name;
    if (__asprintf (&shlib_name, "libnss_%s.so%s",
                    module->name, __nss_shlib_revision) < 0)
      /* This is definitely a temporary failure.  Do not update
         module->state.  This will trigger another attempt at the next
         call.  */
      return false;

    handle = __libc_dlopen (shlib_name);
    free (shlib_name);
  }

  /* Failing to load the module can be caused by several different
     scenarios.  One such scenario is that the module has been removed
     from the disk.  In which case the in-memory version is all that
     we have, and if the module->state indidates it is loaded then we
     can use it.  */
  if (handle == NULL)
    {
      /* dlopen failure.  We do not know if this a temporary or
         permanent error.  See bug 22041.  Update the state using the
         double-checked locking idiom.  */

      __libc_lock_lock (nss_module_list_lock);
      bool result = result;
      switch ((enum nss_module_state) atomic_load_acquire (&module->state))
        {
        case nss_module_uninitialized:
          atomic_store_release (&module->state, nss_module_failed);
          result = false;
          break;
        case nss_module_loaded:
          result = true;
          break;
        case nss_module_failed:
          result = false;
          break;
        }
      __libc_lock_unlock (nss_module_list_lock);
      return result;
    }

  nss_module_functions_untyped pointers;

  /* Look up and store locally all the function pointers we may need
     later.  Doing this now means the data will not change in the
     future.  */
  for (size_t idx = 0; idx < array_length (nss_function_name_array); ++idx)
    {
      char *function_name;
      if (__asprintf (&function_name, "_nss_%s_%s",
                      module->name, nss_function_name_array[idx]) < 0)
        {
          /* Definitely a temporary error.  */
          __libc_dlclose (handle);
          return false;
        }
      pointers[idx] = __libc_dlsym (handle, function_name);
      free (function_name);
#ifdef PTR_MANGLE
      PTR_MANGLE (pointers[idx]);
#endif
    }

# ifdef USE_NSCD
  if (is_nscd)
    {
      /* Call the init function when nscd is used.  */
      size_t initlen = (5 + strlen (module->name)
			+ strlen ("_init") + 1);
      char init_name[initlen];

      /* Construct the init function name.  */
      __stpcpy (__stpcpy (__stpcpy (init_name,
				    "_nss_"),
			  module->name),
		"_init");

      /* Find the optional init function.  */
      void (*ifct) (void (*) (size_t, struct traced_file *))
	= __libc_dlsym (handle, init_name);
      if (ifct != NULL)
	{
	  void (*cb) (size_t, struct traced_file *) = nscd_init_cb;
#  ifdef PTR_DEMANGLE
	  PTR_DEMANGLE (cb);
#  endif
	  ifct (cb);
	}
    }
# endif

  /* Install the function pointers, following the double-checked
     locking idiom.  Delay this after all processing, in case loading
     the module triggers unwinding.  */
  __libc_lock_lock (nss_module_list_lock);
  switch ((enum nss_module_state) atomic_load_acquire (&module->state))
    {
    case nss_module_uninitialized:
    case nss_module_failed:
      memcpy (module->functions.untyped, pointers,
              sizeof (module->functions.untyped));
      module->handle = handle;
      /* Synchronizes with unlocked __nss_module_load atomic_load_acquire.  */
      atomic_store_release (&module->state, nss_module_loaded);
      break;
    case nss_module_loaded:
      /* If the module was already loaded, close our own handle.  This
         does not actually unload the modules, only the reference
         counter is decremented for the loaded module.  */
      __libc_dlclose (handle);
      break;
    }
  __libc_lock_unlock (nss_module_list_lock);
  return true;
}

/* Force the module identified by MODULE to be loaded.  We return
   false if the module could not be loaded, true otherwise.  Loading
   the module requires looking up all the possible interface APIs and
   caching the results.  */
bool
__nss_module_load (struct nss_module *module)
{
  switch ((enum nss_module_state) atomic_load_acquire (&module->state))
    {
    case nss_module_uninitialized:
      return module_load (module);
    case nss_module_loaded:
      /* Loading has already succeeded.  */
      return true;
    case nss_module_failed:
      /* Loading previously failed.  */
      return false;
    }
  __builtin_unreachable ();
}

static int
name_search (const void *left, const void *right)
{
  return strcmp (left, right);
}

/* Load module MODULE (if it isn't already) and return a pointer to
   the module's implementation of NAME, otherwise return NULL on
   failure or error.  */
void *
__nss_module_get_function (struct nss_module *module, const char *name)
{
  if (!__nss_module_load (module))
    return NULL;

  function_name *name_entry = bsearch (name, nss_function_name_array,
                                       array_length (nss_function_name_array),
                                       sizeof (function_name), name_search);
  assert (name_entry != NULL);
  size_t idx = name_entry - nss_function_name_array;
  void *fptr = module->functions.untyped[idx];
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (fptr);
#endif
  return fptr;
}

#if defined SHARED && defined USE_NSCD
/* Load all libraries for the service.  */
static void
nss_load_all_libraries (enum nss_database service)
{
  nss_action_list ni = NULL;

  if (__nss_database_get (service, &ni))
    while (ni->module != NULL)
      {
        __nss_module_load (ni->module);
        ++ni;
      }
}

define_traced_file (pwd, _PATH_NSSWITCH_CONF);
define_traced_file (grp, _PATH_NSSWITCH_CONF);
define_traced_file (hst, _PATH_NSSWITCH_CONF);
define_traced_file (serv, _PATH_NSSWITCH_CONF);
define_traced_file (netgr, _PATH_NSSWITCH_CONF);

/* Called by nscd and nscd alone.  */
void
__nss_disable_nscd (void (*cb) (size_t, struct traced_file *))
{
  void (*cb1) (size_t, struct traced_file *);
  cb1 = cb;
# ifdef PTR_MANGLE
  PTR_MANGLE (cb);
# endif
  nscd_init_cb = cb;
  is_nscd = true;

  /* Find all the relevant modules so that the init functions are called.  */
  nss_load_all_libraries (nss_database_passwd);
  nss_load_all_libraries (nss_database_group);
  nss_load_all_libraries (nss_database_hosts);
  nss_load_all_libraries (nss_database_services);

  /* Make sure NSCD purges its cache if nsswitch.conf changes.  */
  init_traced_file (&pwd_traced_file.file, _PATH_NSSWITCH_CONF, 0);
  cb1 (pwddb, &pwd_traced_file.file);
  init_traced_file (&grp_traced_file.file, _PATH_NSSWITCH_CONF, 0);
  cb1 (grpdb, &grp_traced_file.file);
  init_traced_file (&hst_traced_file.file, _PATH_NSSWITCH_CONF, 0);
  cb1 (hstdb, &hst_traced_file.file);
  init_traced_file (&serv_traced_file.file, _PATH_NSSWITCH_CONF, 0);
  cb1 (servdb, &serv_traced_file.file);
  init_traced_file (&netgr_traced_file.file, _PATH_NSSWITCH_CONF, 0);
  cb1 (netgrdb, &netgr_traced_file.file);

  /* Disable all uses of NSCD.  */
  __nss_not_use_nscd_passwd = -1;
  __nss_not_use_nscd_group = -1;
  __nss_not_use_nscd_hosts = -1;
  __nss_not_use_nscd_services = -1;
  __nss_not_use_nscd_netgroup = -1;
}
#endif

/* Block attempts to dlopen any module we haven't already opened.  */
void
__nss_module_disable_loading (void)
{
  __libc_lock_lock (nss_module_list_lock);

  for (struct nss_module *p = nss_module_list; p != NULL; p = p->next)
    if (p->state == nss_module_uninitialized)
      p->state = nss_module_failed;

  __libc_lock_unlock (nss_module_list_lock);
}

void __libc_freeres_fn_section
__nss_module_freeres (void)
{
  struct nss_module *current = nss_module_list;
  while (current != NULL)
    {
      /* Ignore built-in modules (which have a NULL handle).  */
      if (current->state == nss_module_loaded
	  && current->handle != NULL)
        __libc_dlclose (current->handle);

      struct nss_module *next = current->next;
      free (current);
      current = next;
    }
  nss_module_list = NULL;
}
