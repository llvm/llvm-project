/* Convert an address family to a string.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/format_nss.h>

#include <support/support.h>

char *
support_format_address_family (int family)
{
  switch (family)
    {
    case AF_INET:
      return xstrdup ("INET");
    case AF_INET6:
      return xstrdup ("INET6");
    case AF_LOCAL:
      return xstrdup ("LOCAL");
    case AF_UNSPEC:
      return xstrdup ("UNSPEC");
    default:
      return xasprintf ("<unknown address family %d>", family);
    }
}
