/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mark Kettenis <kettenis@phys.uva.nl>, 1998.

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

#include <string.h>
#include <unistd.h>

#define TRANSFORM_UTMP_FILE_NAME(file_name)		\
  ((strcmp (file_name, _PATH_UTMP) == 0			\
    && __access (_PATH_UTMP "x", F_OK) == 0)		\
   ? (_PATH_UTMP "x")					\
   : ((strcmp (file_name, _PATH_WTMP) == 0		\
       && __access ( _PATH_WTMP "x", F_OK) == 0)	\
      ? (_PATH_WTMP "x")				\
      : ((strcmp (file_name, _PATH_UTMP "x") == 0	\
	  && __access (_PATH_UTMP "x", F_OK) != 0)	\
	 ? _PATH_UTMP					\
	 : ((strcmp (file_name, _PATH_WTMP "x") == 0	\
	     && __access (_PATH_WTMP "x", F_OK) != 0)	\
	    ? _PATH_WTMP				\
	    : file_name))))

#include <login/updwtmp.c>
