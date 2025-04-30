/* String tables for ld.so.cache construction.  Deallocation (for tests only).
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <stdlib.h>
#include <stringtable.h>

void
stringtable_free (struct stringtable *table)
{
  for (uint32_t i = 0; i < table->allocated; ++i)
    for (struct stringtable_entry *e = table->entries[i]; e != NULL; )
      {
        struct stringtable_entry *next = e->next;
        free (e);
        e = next;
      }
  free (table->entries);
  *table = (struct stringtable) { 0, };
}
