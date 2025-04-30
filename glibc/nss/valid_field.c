/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <nss.h>
#include <string.h>

const char __nss_invalid_field_characters[] = NSS_INVALID_FIELD_CHARACTERS;

/* Check that VALUE is either NULL or a NUL-terminated string which
   does not contain characters not permitted in NSS database
   fields.  */
_Bool
__nss_valid_field (const char *value)
{
  return value == NULL
    || strpbrk (value, __nss_invalid_field_characters) == NULL;
}
