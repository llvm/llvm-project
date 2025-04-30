/* List POSIX compilation environments for this libc.
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

#include <unistd.h>

#define START_ENV_GROUP(VERSION)		\
  /* Empty.  */

#define END_ENV_GROUP(VERSION)			\
  /* Empty.  */

#define KNOWN_ABSENT_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)	\
  /* Empty.  */

#define KNOWN_PRESENT_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)	\
  @@@PRESENT_##ENV_PREFIX##_##SUFFIX

#define UNKNOWN_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)	\
  /* Empty.  */

#include "posix-envs.def"

#undef START_ENV_GROUP
#undef END_ENV_GROUP
#undef KNOWN_ABSENT_ENVIRONMENT
#undef KNOWN_PRESENT_ENVIRONMENT
#undef UNKNOWN_ENVIRONMENT
