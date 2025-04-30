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

/* Rewrite VALUE to a valid field value in the NSS database.  Invalid
   characters are replaced with a single space character ' '.  If
   VALUE is NULL, the empty string is returned.  *TO_BE_FREED is
   overwritten with a pointer the caller has to free if the function
   returns successfully.  On failure, return NULL.  */
const char *
__nss_rewrite_field (const char *value, char **to_be_freed)
{
  *to_be_freed = NULL;
  if (value == NULL)
    return "";

  /* Search for non-allowed characters.  */
  const char *p = strpbrk (value, __nss_invalid_field_characters);
  if (p == NULL)
    return value;
  *to_be_freed = __strdup (value);
  if (*to_be_freed == NULL)
    return NULL;

  /* Switch pointer to freshly-allocated buffer.  */
  char *bad = *to_be_freed + (p - value);
  do
    {
      *bad = ' ';
      bad = strpbrk (bad + 1, __nss_invalid_field_characters);
    }
  while (bad != NULL);

  return *to_be_freed;
}
