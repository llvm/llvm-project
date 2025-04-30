/* Configuration for localedef program.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1995.

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

#ifndef _LD_CONFIG_H
#define _LD_CONFIG_H	1

/* Use the internal textdomain used for libc messages.  */
#define PACKAGE _libc_intl_domainname
#ifndef VERSION
/* Get libc version number.  */
#include "../../version.h"
#endif

#define DEFAULT_CHARMAP "ANSI_X3.4-1968" /* ASCII */

/* This must be one higer than the last used LC_xxx category value.  */
#define __LC_LAST	13

#include_next <config.h>
#endif
