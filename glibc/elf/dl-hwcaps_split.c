/* Hardware capability support for run-time dynamic loader.  String splitting.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <dl-hwcaps.h>
#include <stdbool.h>
#include <string.h>

_Bool
_dl_hwcaps_split (struct dl_hwcaps_split *s)
{
  if (s->segment == NULL)
    return false;

  /* Skip over the previous segment.   */
  s->segment += s->length;

  /* Consume delimiters.  This also avoids returning an empty
     segment.  */
  while (*s->segment == ':')
    ++s->segment;
  if (*s->segment == '\0')
    return false;

  /* This could use strchrnul, but we would have to link the function
     into ld.so for that.  */
  const char *colon = strchr (s->segment, ':');
  if (colon == NULL)
    s->length = strlen (s->segment);
  else
    s->length = colon - s->segment;
  return true;
}

_Bool
_dl_hwcaps_split_masked (struct dl_hwcaps_split_masked *s)
{
  while (true)
    {
      if (!_dl_hwcaps_split (&s->split))
        return false;
      bool active = s->bitmask & 1;
      s->bitmask >>= 1;
      if (active && _dl_hwcaps_contains (s->mask,
                                         s->split.segment, s->split.length))
        return true;
    }
}

_Bool
_dl_hwcaps_contains (const char *hwcaps, const char *name, size_t name_length)
{
  if (hwcaps == NULL)
    return true;

  struct dl_hwcaps_split split;
  _dl_hwcaps_split_init (&split, hwcaps);
  while (_dl_hwcaps_split (&split))
    if (split.length == name_length
        && memcmp (split.segment, name, name_length) == 0)
      return true;
  return false;
}
