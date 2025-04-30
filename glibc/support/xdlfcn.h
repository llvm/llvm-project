/* Support functionality for using dlopen/dlclose/dlsym.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_DLOPEN_H
#define SUPPORT_DLOPEN_H

#include <dlfcn.h>

__BEGIN_DECLS

/* Each of these terminates process on failure with relevant error message.  */
void *xdlopen (const char *filename, int flags);
void *xdlmopen (Lmid_t lmid, const char *filename, int flags);
void *xdlsym (void *handle, const char *symbol);
void *xdlvsym (void *handle, const char *symbol, const char *version);
void xdlclose (void *handle);

__END_DECLS

#endif /* SUPPORT_DLOPEN_H */
