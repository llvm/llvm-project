/* Close a handle opened by `dlopen'.
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

int
__dlclose (void *handle)
{
#ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->dlclose (handle);
#endif

  return _dlerror_run (GLRO (dl_close), handle) ? -1 : 0;
}
versioned_symbol (libc, __dlclose, dlclose, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libdl, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libdl, __dlclose, dlclose, GLIBC_2_0);
#endif
