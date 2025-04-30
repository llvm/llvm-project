/* Example for creating a struct dirent object for use with glob.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#include <dirent.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

struct dirent *
mkdirent (const char *name)
{
  size_t dirent_size = offsetof (struct dirent, d_name) + 1;
  size_t name_length = strlen (name);
  size_t total_size = dirent_size + name_length;
  if (total_size < dirent_size)
    {
      errno = ENOMEM;
      return NULL;
    }
  struct dirent *result = malloc (total_size);
  if (result == NULL)
    return NULL;
  result->d_type = DT_UNKNOWN;
  result->d_ino = 1;            /* Do not skip this entry.  */
  memcpy (result->d_name, name, name_length + 1);
  return result;
}
