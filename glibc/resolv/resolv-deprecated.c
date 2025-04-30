/* Forwarders for deprecated libresolv functions which are implemented in libc.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* Some functions are used by the stub resolver implementation
   internally and thus have to be located in libc.  They have been
   historially exported for application use as well.  However, the
   stub resolver dependency on those functions is not intrinsic to
   what the stub resolver does, and it is possible that a future
   version of the stub resolver will not need them anymore.  The
   public symbols for these functions remain in libresolv, and are not
   moved to libc, to avoid adding new symbol versions for legacy
   functions.  */

#include <arpa/nameser.h>
#include <resolv.h>

int
ns_makecanon (const char *src, char *dst, size_t dstsize)
{
  return __libc_ns_makecanon (src, dst, dstsize);
}

int
ns_samename (const char *a, const char *b)
{
  return __libc_ns_samename (a, b);
}

int
res_nameinquery (const char *name, int type, int class,
                 const unsigned char *buf, const unsigned char *eom)
{
  return __libc_res_nameinquery (name, type, class, buf, eom);
}

int
res_queriesmatch (const unsigned char *buf1, const unsigned char *eom1,
                  const unsigned char *buf2, const unsigned char *eom2)
{
  return __libc_res_queriesmatch (buf1, eom1, buf2, eom2);
}
