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

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <netdb.h>
#include <libc-lock.h>
#include <search.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>

#include <aliases.h>
#include <grp.h>
#include <netinet/ether.h>
#include <pwd.h>
#include <shadow.h>
#include <unistd.h>

#if !defined DO_STATIC_NSS || defined SHARED
# include <gnu/lib-names.h>
#endif

#include "nsswitch.h"
#include "../nscd/nscd_proto.h"
#include <sysdep.h>
#include <config.h>

/* Declare external database variables.  */
#define DEFINE_DATABASE(name)						      \
  nss_action_list __nss_##name##_database attribute_hidden;		      \
  weak_extern (__nss_##name##_database)
#include "databases.def"
#undef DEFINE_DATABASE


#ifdef USE_NSCD
/* Flags whether custom rules for database is set.  */
bool __nss_database_custom[NSS_DBSIDX_max];
#endif

/*__libc_lock_define_initialized (static, lock)*/

/* -1 == not found
    0 == function found
    1 == finished */
int
__nss_lookup (nss_action_list *ni, const char *fct_name, const char *fct2_name,
	      void **fctp)
{
  *fctp = __nss_lookup_function (*ni, fct_name);
  if (*fctp == NULL && fct2_name != NULL)
    *fctp = __nss_lookup_function (*ni, fct2_name);

  while (*fctp == NULL
	 && nss_next_action (*ni, NSS_STATUS_UNAVAIL) == NSS_ACTION_CONTINUE
	 && (*ni)[1].module != NULL)
    {
      ++(*ni);

      *fctp = __nss_lookup_function (*ni, fct_name);
      if (*fctp == NULL && fct2_name != NULL)
	*fctp = __nss_lookup_function (*ni, fct2_name);
    }

  return *fctp != NULL ? 0 : (*ni)[1].module == NULL ? 1 : -1;
}
libc_hidden_def (__nss_lookup)


/* -1 == not found
    0 == adjusted for next function
    1 == finished */
int
__nss_next2 (nss_action_list *ni, const char *fct_name, const char *fct2_name,
	     void **fctp, int status, int all_values)
{
  if (all_values)
    {
      if (nss_next_action (*ni, NSS_STATUS_TRYAGAIN) == NSS_ACTION_RETURN
	  && nss_next_action (*ni, NSS_STATUS_UNAVAIL) == NSS_ACTION_RETURN
	  && nss_next_action (*ni, NSS_STATUS_NOTFOUND) == NSS_ACTION_RETURN
	  && nss_next_action (*ni, NSS_STATUS_SUCCESS) == NSS_ACTION_RETURN)
	return 1;
    }
  else
    {
      /* This is really only for debugging.  */
      if (__builtin_expect (NSS_STATUS_TRYAGAIN > status
			    || status > NSS_STATUS_RETURN, 0))
	 __libc_fatal ("Illegal status in __nss_next.\n");

       if (nss_next_action (*ni, status) == NSS_ACTION_RETURN)
	 return 1;
    }

  if ((*ni)[1].module == NULL)
    return -1;

  do
    {
      ++(*ni);

      *fctp = __nss_lookup_function (*ni, fct_name);
      if (*fctp == NULL && fct2_name != NULL)
	*fctp = __nss_lookup_function (*ni, fct2_name);
    }
  while (*fctp == NULL
	 && nss_next_action (*ni, NSS_STATUS_UNAVAIL) == NSS_ACTION_CONTINUE
	 && (*ni)[1].module != NULL);

  return *fctp != NULL ? 0 : -1;
}
libc_hidden_def (__nss_next2)

void *
__nss_lookup_function (nss_action_list ni, const char *fct_name)
{
  if (ni->module == NULL)
    return NULL;
  return __nss_module_get_function (ni->module, fct_name);
}
libc_hidden_def (__nss_lookup_function)
