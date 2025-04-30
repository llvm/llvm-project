/* Specify whether there should be compat symbol aliases for some
   classification functions.  Generic version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _LDBL_CLASSIFY_COMPAT_H
#define _LDBL_CLASSIFY_COMPAT_H	1

/* If defined to 1, enable __finitel, __isinfl, and __isnanl function
   aliases for binary compatibility when built without long double
   support.  If defined to 0, or if long double does not have the same
   format as double, there are no such aliases.  New ports should use
   the default definition of this as 0, as such
   implementation-namespace functions should only have one exported
   name per floating-point format, not one per floating-point
   type.  */
#define LDBL_CLASSIFY_COMPAT 0

#endif /* ldbl-classify-compat.h */
