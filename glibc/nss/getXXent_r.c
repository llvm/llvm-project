/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <errno.h>
#include <libc-lock.h>

#include "nsswitch.h"

/*******************************************************************\
|* Here we assume several symbols to be defined:		   *|
|* 								   *|
|* LOOKUP_TYPE   - the return type of the function		   *|
|* 								   *|
|* SETFUNC_NAME  - name of the non-reentrant setXXXent function	   *|
|* 								   *|
|* GETFUNC_NAME  - name of the non-reentrant getXXXent function	   *|
|* 								   *|
|* ENDFUNC_NAME  - name of the non-reentrant endXXXent function	   *|
|* 								   *|
|* DATABASE_NAME - name of the database the function accesses	   *|
|*		   (e.g., host, services, ...)			   *|
|* 								   *|
|* Optionally the following vars can be defined:		   *|
|* 								   *|
|* STAYOPEN      - variable declaration for setXXXent function	   *|
|* 								   *|
|* STAYOPEN_VAR  - variable name for setXXXent function		   *|
|* 								   *|
|* NEED_H_ERRNO  - an extra parameter will be passed to point to   *|
|*		   the global `h_errno' variable.		   *|
|* 								   *|
\*******************************************************************/

/* To make the real sources a bit prettier.  */
#define REENTRANT_GETNAME APPEND_R (GETFUNC_NAME)
#define APPEND_R(Name) CONCAT2_2 (Name, _r)
#define INTERNAL(Name) CONCAT2_2 (__, Name)
#define CONCAT2_1(Pre, Post) CONCAT2_2 (Pre, Post)
#define CONCAT2_2(Pre, Post) Pre##Post
#define NEW(name) NEW1 (name)
#define NEW1(name) __new_##name

#define SETFUNC_NAME_STRING STRINGIZE (SETFUNC_NAME)
#define GETFUNC_NAME_STRING STRINGIZE (REENTRANT_GETNAME)
#define ENDFUNC_NAME_STRING STRINGIZE (ENDFUNC_NAME)
#define DATABASE_NAME_STRING STRINGIZE (DATABASE_NAME)
#define STRINGIZE(Name) STRINGIZE1 (Name)
#define STRINGIZE1(Name) #Name

#ifndef DB_LOOKUP_FCT
# define DB_LOOKUP_FCT CONCAT3_1 (__nss_, DATABASE_NAME, _lookup2)
# define CONCAT3_1(Pre, Name, Post) CONCAT3_2 (Pre, Name, Post)
# define CONCAT3_2(Pre, Name, Post) Pre##Name##Post
#endif

/* Sometimes we need to store error codes in the `h_errno' variable.  */
#ifdef NEED_H_ERRNO
# define H_ERRNO_PARM , int *h_errnop
# define H_ERRNO_VAR , &h_errno
# define H_ERRNO_VAR_P &h_errno
#else
# define H_ERRNO_PARM
# define H_ERRNO_VAR
# define H_ERRNO_VAR_P NULL
#endif

/* Some databases take the `stayopen' flag.  */
#ifdef STAYOPEN
# define STAYOPEN_TMP CONCAT2_1 (STAYOPEN, _tmp)
# define STAYOPEN_TMPVAR &CONCAT2_1 (STAYOPEN_VAR, _tmp)
#else
# define STAYOPEN void
# define STAYOPEN_VAR 0
# define STAYOPEN_TMPVAR NULL
#endif

#ifndef NEED__RES
# define NEED__RES 0
#endif

/* This handle for the NSS data base is shared between all
   set/get/endXXXent functions.  */
static nss_action_list nip;
/* Remember the last service used since the last call to  `endXXent'.  */
static nss_action_list last_nip;
/* Remember the first service_entry across set/get/endent.  */
static nss_action_list startp;

#ifdef STAYOPEN_TMP
/* We need to remember the last `stayopen' flag given by the user
   since the `setent' function is only called for the first available
   service.  */
static STAYOPEN_TMP;
#endif

/* Protect above variable against multiple uses at the same time.  */
__libc_lock_define_initialized (static, lock)

/* The lookup function for the first entry of this service.  */
extern int DB_LOOKUP_FCT (nss_action_list *nip, const char *name,
			  const char *name2, void **fctp);
libc_hidden_proto (DB_LOOKUP_FCT)

void
SETFUNC_NAME (STAYOPEN)
{
  int save;

  __libc_lock_lock (lock);
  __nss_setent (SETFUNC_NAME_STRING, DB_LOOKUP_FCT, &nip, &startp,
		&last_nip, STAYOPEN_VAR, STAYOPEN_TMPVAR, NEED__RES);

  save = errno;
  __libc_lock_unlock (lock);
  __set_errno (save);
}


void
ENDFUNC_NAME (void)
{
  int save;

  /* If the service has not been used before do not do anything.  */
  if (startp != NULL)
    {
      __libc_lock_lock (lock);
      __nss_endent (ENDFUNC_NAME_STRING, DB_LOOKUP_FCT, &nip, &startp,
		    &last_nip, NEED__RES);
      save = errno;
      __libc_lock_unlock (lock);
      __set_errno (save);
    }
}


int
INTERNAL (REENTRANT_GETNAME) (LOOKUP_TYPE *resbuf, char *buffer, size_t buflen,
			      LOOKUP_TYPE **result H_ERRNO_PARM)
{
  int status;
  int save;

  __libc_lock_lock (lock);
  status = __nss_getent_r (GETFUNC_NAME_STRING, SETFUNC_NAME_STRING,
			   DB_LOOKUP_FCT, &nip, &startp, &last_nip,
			   STAYOPEN_TMPVAR, NEED__RES, resbuf, buffer,
			   buflen, (void **) result, H_ERRNO_VAR_P);
  save = errno;
  __libc_lock_unlock (lock);
  __set_errno (save);
  return status;
}


#ifdef NO_COMPAT_NEEDED
strong_alias (INTERNAL (REENTRANT_GETNAME), REENTRANT_GETNAME);
#else
# include <shlib-compat.h>
# if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1_2)
#  define OLD(name) OLD1 (name)
#  define OLD1(name) __old_##name

int
attribute_compat_text_section
OLD (REENTRANT_GETNAME) (LOOKUP_TYPE *resbuf, char *buffer, size_t buflen,
			 LOOKUP_TYPE **result H_ERRNO_PARM)
{
  int ret = INTERNAL (REENTRANT_GETNAME) (resbuf, buffer, buflen,
					  result H_ERRNO_VAR);

  if (ret != 0)
    ret = -1;

  return ret;
}

#  define do_symbol_version(real, name, version) \
  compat_symbol (libc, real, name, version)
do_symbol_version (OLD (REENTRANT_GETNAME), REENTRANT_GETNAME, GLIBC_2_0);
# endif

/* As INTERNAL (REENTRANT_GETNAME) may be hidden, we need an alias
   in between so that the REENTRANT_GETNAME@@GLIBC_2.1.2 is not
   hidden too.  */
strong_alias (INTERNAL (REENTRANT_GETNAME), NEW (REENTRANT_GETNAME));

# define do_default_symbol_version(real, name, version) \
  versioned_symbol (libc, real, name, version)
do_default_symbol_version (NEW (REENTRANT_GETNAME),
			   REENTRANT_GETNAME, GLIBC_2_1_2);
#endif

nss_interface_function (SETFUNC_NAME)
nss_interface_function (ENDFUNC_NAME)
nss_interface_function (REENTRANT_GETNAME)
