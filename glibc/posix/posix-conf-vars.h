/* Macros to check if a POSIX configuration variable is defined or set.

   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef _POSIX_CONF_VARS_H
#define _POSIX_CONF_VARS_H

/* The script gen-posix-conf-vars.awk generates the header
   posix-conf-vars-def.h from the list file posix-conf-vars.list and defines
   CONF_DEF_* macros for each entry in the list file set to either of
   CONF_DEF_UNDEFINED, CONF_DEF_DEFINED_SET or CONF_DEF_DEFINED_UNSET.  To
   check configuration variables within glibc code, use the configuration macro
   functions instead of checking for definitions of the macros.  */

#include <posix-conf-vars-def.h>

#define CONF_DEF_UNDEFINED	1
#define CONF_DEF_DEFINED_SET	2
#define CONF_DEF_DEFINED_UNSET	3

/* The configuration variable is not defined.  */
#define CONF_IS_UNDEFINED(conf) (CONF_DEF##conf == CONF_DEF_UNDEFINED)

/* The configuration variable is defined.  It may or may not be set.  */
#define CONF_IS_DEFINED(conf) (CONF_DEF##conf != CONF_DEF_UNDEFINED)

/* The configuration variable is defined and set.  */
#define CONF_IS_DEFINED_SET(conf) (CONF_DEF##conf == CONF_DEF_DEFINED_SET)

/* The configuration variable is defined but not set.  */
#define CONF_IS_DEFINED_UNSET(conf) (CONF_DEF##conf == CONF_DEF_DEFINED_UNSET)

#endif
