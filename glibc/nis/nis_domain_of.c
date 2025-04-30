/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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

#include <rpcsvc/nis.h>
#include <shlib-compat.h>

nis_name
nis_domain_of (const_nis_name name)
{
  static char result[NIS_MAXNAMELEN + 1];

  return nis_domain_of_r (name, result, NIS_MAXNAMELEN);
}
libnsl_hidden_nolink_def (nis_domain_of, GLIBC_2_1)

const_nis_name
__nis_domain_of (const_nis_name name)
{
  const_nis_name cptr = strchr (name, '.');

  if (cptr == NULL)
    return "";

  if (*++cptr == '\0')
    return ".";

  return cptr;
}
