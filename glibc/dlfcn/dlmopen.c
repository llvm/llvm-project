/* Load a shared object at run time.
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
#include <errno.h>
#include <libintl.h>
#include <stddef.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <shlib-compat.h>

struct dlmopen_args
{
  /* Namespace ID.  */
  Lmid_t nsid;
  /* The arguments for dlopen_doit.  */
  const char *file;
  int mode;
  /* The return value of dlopen_doit.  */
  void *new;
  /* Address of the caller.  */
  const void *caller;
};

static void
dlmopen_doit (void *a)
{
  struct dlmopen_args *args = (struct dlmopen_args *) a;

  /* Non-shared code has no support for multiple namespaces.  */
  if (args->nsid != LM_ID_BASE)
    {
# ifdef SHARED
      /* If trying to open the link map for the main executable the namespace
	 must be the main one.  */
      if (args->file == NULL)
# endif
	_dl_signal_error (EINVAL, NULL, NULL, N_("invalid namespace"));

      /* It makes no sense to use RTLD_GLOBAL when loading a DSO into
	 a namespace other than the base namespace.  */
      if (__glibc_unlikely (args->mode & RTLD_GLOBAL))
	_dl_signal_error (EINVAL, NULL, NULL, N_("invalid mode"));
    }

  args->new = GLRO(dl_open) (args->file ?: "", args->mode | __RTLD_DLOPEN,
			     args->caller,
			     args->nsid, __libc_argc, __libc_argv, __environ);
}

static void *
dlmopen_implementation (Lmid_t nsid, const char *file, int mode,
			void *dl_caller)
{
  struct dlmopen_args args;
  args.nsid = nsid;
  args.file = file;
  args.mode = mode;
  args.caller = dl_caller;

  return _dlerror_run (dlmopen_doit, &args) ? NULL : args.new;
}

#ifdef SHARED
void *
___dlmopen (Lmid_t nsid, const char *file, int mode)
{
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->dlmopen (nsid, file, mode, RETURN_ADDRESS (0));
  else
    return dlmopen_implementation (nsid, file, mode, RETURN_ADDRESS (0));
}
versioned_symbol (libc, ___dlmopen, dlmopen, GLIBC_2_34);

# if OTHER_SHLIB_COMPAT (libdl, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (libdl, ___dlmopen, dlmopen, GLIBC_2_3_4);
# endif
#else /* !SHARED */
/* Also used with _dlfcn_hook.  */
void *
__dlmopen (Lmid_t nsid, const char *file, int mode, void *dl_caller)
{
  return dlmopen_implementation (nsid, file, mode, RETURN_ADDRESS (0));
}

void *
___dlmopen (Lmid_t nsid, const char *file, int mode)
{
  return __dlmopen (nsid, file, mode, RETURN_ADDRESS (0));
}
weak_alias (___dlmopen, dlmopen)
static_link_warning (dlmopen)
#endif /* !SHARED */
