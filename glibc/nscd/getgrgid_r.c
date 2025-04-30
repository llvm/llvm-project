/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <grp.h>

#include <grp-merge.h>

#define LOOKUP_TYPE	struct group
#define FUNCTION_NAME	getgrgid
#define DATABASE_NAME	group
#define ADD_PARAMS	gid_t gid
#define ADD_VARIABLES	gid
#define BUFLEN		NSS_BUFLEN_GROUP

#define DEEPCOPY_FN	__copy_grp
#define MERGE_FN	__merge_grp

/* We are nscd, so we don't want to be talking to ourselves.  */
#undef	USE_NSCD

#include <nss/getXXbyYY_r.c>
