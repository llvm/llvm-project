/* dlinfo -- Get information from the dynamic linker.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <link.h>
#include <ldsodefs.h>
#include <libintl.h>
#include <dl-tls.h>
#include <shlib-compat.h>

struct dlinfo_args
{
  void *handle;
  int request;
  void *arg;
};

static void
dlinfo_doit (void *argsblock)
{
  struct dlinfo_args *const args = argsblock;
  struct link_map *l = args->handle;
  struct pthread *pd;

  switch (args->request)
    {
    case RTLD_DI_CONFIGADDR:
    default:
      _dl_signal_error (0, NULL, NULL, N_("unsupported dlinfo request"));
      break;

    case RTLD_DI_LMID:
      *(Lmid_t *) args->arg = l->l_ns;
      break;

    case RTLD_DI_LINKMAP:
      *(struct link_map **) args->arg = l;
      break;

    case RTLD_DI_SERINFO:
      _dl_rtld_di_serinfo (l, args->arg, false);
      break;
    case RTLD_DI_SERINFOSIZE:
      _dl_rtld_di_serinfo (l, args->arg, true);
      break;

    case RTLD_DI_ORIGIN:
      strcpy (args->arg, l->l_origin);
      break;

    case RTLD_DI_TLS_MODID:
      *(size_t *) args->arg = 0;
      *(size_t *) args->arg = l->l_tls_modid;
      break;

    case RTLD_DI_TLS_DATA:
      {
	void *data = NULL;
	if (l->l_tls_modid != 0)
	  data = GLRO(dl_tls_get_addr_soft) (l, THREAD_SELF);
	*(void **) args->arg = data;
	break;
      }

    case RTLD_DI_GENERIC_TLS_DATA:
      {
	void *data = NULL;
	if (l->l_tls_modid != 0) {
	  pd = *(struct pthread **) args->arg;
	  data = GLRO(dl_tls_get_addr_soft) (l, pd);
	}
	*(void **) args->arg = data;
	break;
      }
    }
}

static int
dlinfo_implementation (void *handle, int request, void *arg)
{
  struct dlinfo_args args = { handle, request, arg };
  return _dlerror_run (&dlinfo_doit, &args) ? -1 : 0;
}

#ifdef SHARED
int
___dlinfo (void *handle, int request, void *arg)
{
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->dlinfo (handle, request, arg);
  else
    return dlinfo_implementation (handle, request, arg);
}
versioned_symbol (libc, ___dlinfo, dlinfo, GLIBC_2_34);

# if OTHER_SHLIB_COMPAT (libdl, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (libc, ___dlinfo, dlinfo, GLIBC_2_3_3);
# endif
#else /* !SHARED */
/* Also used with _dlfcn_hook.  */
int
__dlinfo (void *handle, int request, void *arg)
{
  return dlinfo_implementation (handle, request, arg);
}
weak_alias (__dlinfo, dlinfo)
#endif
