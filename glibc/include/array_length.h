/* The array_length and array_end macros.
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

#ifndef _ARRAY_LENGTH_H
#define _ARRAY_LENGTH_H

/* array_length (VAR) is the number of elements in the array VAR.  VAR
   must evaluate to an array, not a pointer.  */
#define array_length(var)                                               \
  (sizeof (var) / sizeof ((var)[0])                                     \
   + 0 * sizeof (struct {                                               \
       _Static_assert (!__builtin_types_compatible_p                    \
                       (__typeof (var), __typeof (&(var)[0])),          \
                       "argument must be an array");                    \
   }))

/* array_end (VAR) is a pointer one past the end of the array VAR.
   VAR must evaluate to an array, not a pointer.  */
#define array_end(var) (&(var)[array_length (var)])

#endif /* _ARRAY_LENGTH_H */
