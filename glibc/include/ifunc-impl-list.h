/* Internal header file for __libc_supported_implementations.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _IFUNC_IMPL_LIST_H
#define _IFUNC_IMPL_LIST_H	1

#include <stdbool.h>
#include <stddef.h>

struct libc_ifunc_impl
{
  /* The name of function to be tested.  */
  const char *name;
  /* The address of function to be tested.  */
  void (*fn) (void);
  /* True if this implementation is usable on this machine.  */
  bool usable;
};

/* Add an IFUNC implementation, IMPL, for function FUNC, to ARRAY with
   USABLE at index I and advance I by one.  */
#define IFUNC_IMPL_ADD(array, i, func, usable, impl) \
  extern __typeof (func) impl attribute_hidden; \
  (array)[i++] = (struct libc_ifunc_impl) { #impl, (void (*) (void)) impl, (usable) };

/* Return the number of IFUNC implementations, N, for function FUNC if
   string NAME matches FUNC.  */
#define IFUNC_IMPL(n, name, func, ...) \
  if (strcmp (name, #func) == 0) \
    { \
      __VA_ARGS__; \
      return n; \
    }

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME and return the number of valid entries.  */
extern size_t __libc_ifunc_impl_list (const char *name,
				      struct libc_ifunc_impl *array,
				      size_t max);

#endif /* ifunc-impl-list.h */
