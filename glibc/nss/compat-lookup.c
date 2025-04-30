/* Compatibility stubs of accidentally exported __nss_*_lookup functions.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <nsswitch.h>

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_27)

# include <errno.h>

/* On i386, the function calling convention changed from the standard
   ABI calling convention to three register parameters in glibc 2.8.
   The following error-returning stub happens to be compatible with
   glibc 2.7 and earlier and glibc 2.8 and later, even on i386.  */
int
attribute_compat_text_section
__nss_passwd_lookup (nss_action_list *ni, const char *fct_name, void **fctp)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, __nss_passwd_lookup, __nss_passwd_lookup, GLIBC_2_0);
strong_alias (__nss_passwd_lookup, __nss_group_lookup)
compat_symbol (libc, __nss_group_lookup, __nss_group_lookup, GLIBC_2_0);
strong_alias (__nss_passwd_lookup, __nss_hosts_lookup)
compat_symbol (libc, __nss_hosts_lookup, __nss_hosts_lookup, GLIBC_2_0);

#endif /* SHLIB_COMPAT */

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_30)

/* These functions were exported under a non-GLIBC_PRIVATE version,
   even though it is not usable externally due to the service_user
   (now nss_action_list) type dependency.  */

int
attribute_compat_text_section
__nss_next (nss_action_list *ni, const char *fct_name, void **fctp, int status,
            int all_values)
{
  return -1;
}
compat_symbol (libc, __nss_next, __nss_next, GLIBC_2_0);

int
attribute_compat_text_section
__nss_database_lookup (const char *database, const char *alternate_name,
                       const char *defconfig, nss_action_list *ni)
{
  *ni = NULL;
  return -1;
}
compat_symbol (libc, __nss_database_lookup, __nss_database_lookup, GLIBC_2_0);

#endif /* SHLIB_COMPAT */
