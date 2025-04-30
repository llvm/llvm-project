/* Look up a symbol in a shared object loaded by `dlopen'.
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

#include <dlfcn.h>
#include <ldsodefs.h>
#include <shlib-compat.h>
#include <stddef.h>

struct dlsym_args
{
  /* The arguments to dlsym_doit.  */
  void *handle;
  const char *name;
  void *who;

  /* The return value of dlsym_doit.  */
  void *sym;
};

static void
dlsym_doit (void *a)
{
  struct dlsym_args *args = (struct dlsym_args *) a;

  args->sym = _dl_sym (args->handle, args->name, args->who);
}

static void *
dlsym_implementation (void *handle, const char *name, void *dl_caller)
{
  struct dlsym_args args;
  args.who = dl_caller;
  args.handle = handle;
  args.name = name;

  /* Protect against concurrent loads and unloads.  */
  __rtld_lock_lock_recursive (GL(dl_load_lock));

  void *result = (_dlerror_run (dlsym_doit, &args) ? NULL : args.sym);

  __rtld_lock_unlock_recursive (GL(dl_load_lock));

  return result;
}

#ifdef SHARED
void *
___dlsym (void *handle, const char *name)
{
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->dlsym (handle, name, RETURN_ADDRESS (0));
  else
    return dlsym_implementation (handle, name, RETURN_ADDRESS (0));
}
versioned_symbol (libc, ___dlsym, dlsym, GLIBC_2_34);

# if OTHER_SHLIB_COMPAT (libdl, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libdl, ___dlsym, dlsym, GLIBC_2_0);
# endif

#else /* !SHARED */
/* Also used with _dlfcn_hook.  */
void *
__dlsym (void *handle, const char *name, void *dl_caller)
{
  return dlsym_implementation (handle, name, dl_caller);
}

void *
___dlsym (void *handle, const char *name)
{
  return __dlsym (handle, name, RETURN_ADDRESS (0));
}
weak_alias (___dlsym, dlsym)
#endif /* !SHARED */
