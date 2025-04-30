/* A type for indices and sizes.
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

#ifndef _IDX_H
#define _IDX_H

/* Get ptrdiff_t.  */
#include <stddef.h>

/* Get PTRDIFF_MAX.  */
#include <stdint.h>

/* The type 'idx_t' holds an (array) index or an (object) size.
   Its implementation promotes to a signed integer type,
   which can hold the values
     0..2^63-1 (on 64-bit platforms) or
     0..2^31-1 (on 32-bit platforms).

   Why a signed integer type?

     * Security: Signed types can be checked for overflow via
       '-fsanitize=undefined', but unsigned types cannot.

     * Comparisons without surprises: ISO C99 § 6.3.1.8 specifies a few
       surprising results for comparisons, such as

           (int) -3 < (unsigned long) 7  =>  false
           (int) -3 < (unsigned int) 7   =>  false
       and on 32-bit machines:
           (long) -3 < (unsigned int) 7  =>  false

       This is surprising because the natural comparison order is by
       value in the realm of infinite-precision signed integers (ℤ).

       The best way to get rid of such surprises is to use signed types
       for numerical integer values, and use unsigned types only for
       bit masks and enums.

   Why not use 'size_t' directly?

     * Because 'size_t' is an unsigned type, and a signed type is better.
       See above.

   Why not use 'ptrdiff_t' directly?

     * Maintainability: When reading and modifying code, it helps to know that
       a certain variable cannot have negative values.  For example, when you
       have a loop

         int n = ...;
         for (int i = 0; i < n; i++) ...

       or

         ptrdiff_t n = ...;
         for (ptrdiff_t i = 0; i < n; i++) ...

       you have to ask yourself "what if n < 0?".  Whereas in

         idx_t n = ...;
         for (idx_t i = 0; i < n; i++) ...

       you know that this case cannot happen.

       Similarly, when a programmer writes

         idx_t = ptr2 - ptr1;

       there is an implied assertion that ptr1 and ptr2 point into the same
       object and that ptr1 <= ptr2.

     * Being future-proof: In the future, range types (integers which are
       constrained to a certain range of values) may be added to C compilers
       or to the C standard.  Several programming languages (Ada, Haskell,
       Common Lisp, Pascal) already have range types.  Such range types may
       help producing good code and good warnings.  The type 'idx_t' could
       then be typedef'ed to a range type that is signed after promotion.  */

/* In the future, idx_t could be typedef'ed to a signed range type.
   The clang "extended integer types", supported in Clang 11 or newer
   <https://clang.llvm.org/docs/LanguageExtensions.html#extended-integer-types>,
   are a special case of range types.  However, these types don't support binary
   operators with plain integer types (e.g. expressions such as x > 1).
   Therefore, they don't behave like signed types (and not like unsigned types
   either).  So, we cannot use them here.  */

/* Use the signed type 'ptrdiff_t'.  */
/* Note: ISO C does not mandate that 'size_t' and 'ptrdiff_t' have the same
   size, but it is so on all platforms we have seen since 1990.  */
typedef ptrdiff_t idx_t;

/* IDX_MAX is the maximum value of an idx_t.  */
#define IDX_MAX PTRDIFF_MAX

/* So far no need has been found for an IDX_WIDTH macro.
   Perhaps there should be another macro IDX_VALUE_BITS that does not
   count the sign bit and is therefore one less than PTRDIFF_WIDTH.  */

#endif /* _IDX_H */
