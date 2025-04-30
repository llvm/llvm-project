/* elision-conf.h: Lock elision  configuration.  Stub version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _ELISION_CONF_H
#define _ELISION_CONF_H 1

/* No elision support by default.  */
#define ENABLE_ELISION_SUPPORT 0

/* Whether __lll_unlock_elision expects a pointer argument to the
   adaptive counter.  Here, an unused arbitrary value.  */
#define ELISION_UNLOCK_NEEDS_ADAPT_COUNT 0

#endif
