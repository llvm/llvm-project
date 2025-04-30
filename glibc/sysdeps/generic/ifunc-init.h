/* IFUNC generic definitions.
   This file is part of the GNU C Library.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

/* These macros are used to implement ifunc selection in C.  To implement
   an ifunc function, foo, which returns the address of __foo_impl1 or
   __foo_impl2:

   #define foo __redirect_foo
   #include <foo.h>
   #undef foo
   #define SYMBOL_NAME foo
   #include <ifunc-init.h>

   extern __typeof (REDIRECT_NAME) OPTIMIZE (impl1) attribute_hidden;
   extern __typeof (REDIRECT_NAME) OPTIMIZE (impl2) attribute_hidden;

   static inline void *
   foo_selector (void)
   {
     if (condition)
      return OPTIMIZE (impl2);

     return OPTIMIZE (impl1);
   }

   libc_ifunc_redirected (__redirect_foo, foo, IFUNC_SELECTOR ());
*/

#define PASTER1(x,y)	x##_##y
#define EVALUATOR1(x,y)	PASTER1 (x,y)
#define PASTER2(x,y)	__##x##_##y
#define EVALUATOR2(x,y)	PASTER2 (x,y)

/* Basically set '__redirect_<symbol>' to use as type definition,
   '__<symbol>_<variant>' as the optimized implementation and
   '<symbol>_ifunc_selector' as the IFUNC selector.  */
#define REDIRECT_NAME	EVALUATOR1 (__redirect, SYMBOL_NAME)
#define OPTIMIZE(name)	EVALUATOR2 (SYMBOL_NAME, name)
#define IFUNC_SELECTOR	EVALUATOR1 (SYMBOL_NAME, ifunc_selector)
