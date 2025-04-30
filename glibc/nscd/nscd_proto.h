/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1998.

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

#ifndef _NSCD_PROTO_H
#define _NSCD_PROTO_H 1

#include <grp.h>
#include <netdb.h>
#include <pwd.h>

/* Interval in which we transfer retry to contact the NSCD.  */
#define NSS_NSCD_RETRY	100

/* Type needed in the interfaces.  */
struct nscd_ai_result;


/* Variables for communication between NSCD handler functions and NSS.  */
extern int __nss_not_use_nscd_passwd attribute_hidden;
extern int __nss_not_use_nscd_group attribute_hidden;
extern int __nss_not_use_nscd_hosts attribute_hidden;
extern int __nss_not_use_nscd_services attribute_hidden;
extern int __nss_not_use_nscd_netgroup attribute_hidden;

extern int __nscd_getpwnam_r (const char *name, struct passwd *resultbuf,
			      char *buffer, size_t buflen,
			      struct passwd **result) attribute_hidden;
extern int __nscd_getpwuid_r (uid_t uid, struct passwd *resultbuf,
			      char *buffer,  size_t buflen,
			      struct passwd **result) attribute_hidden;
extern int __nscd_getgrnam_r (const char *name, struct group *resultbuf,
			      char *buffer, size_t buflen,
			      struct group **result) attribute_hidden;
extern int __nscd_getgrgid_r (gid_t gid, struct group *resultbuf,
			      char *buffer,  size_t buflen,
			      struct group **result) attribute_hidden;
extern int __nscd_gethostbyname_r (const char *name,
				   struct hostent *resultbuf,
				   char *buffer, size_t buflen,
				   struct hostent **result, int *h_errnop)
     attribute_hidden;
extern int __nscd_gethostbyname2_r (const char *name, int af,
				    struct hostent *resultbuf,
				    char *buffer, size_t buflen,
				    struct hostent **result, int *h_errnop)
     attribute_hidden;
extern int __nscd_gethostbyaddr_r (const void *addr, socklen_t len, int type,
				   struct hostent *resultbuf,
				   char *buffer, size_t buflen,
				   struct hostent **result, int *h_errnop)
     attribute_hidden;
extern int __nscd_getai (const char *key, struct nscd_ai_result **result,
			 int *h_errnop) attribute_hidden;
extern int __nscd_getgrouplist (const char *user, gid_t group, long int *size,
				gid_t **groupsp, long int limit)
     attribute_hidden;
extern int __nscd_getservbyname_r (const char *name, const char *proto,
				   struct servent *result_buf, char *buf,
				   size_t buflen, struct servent **result)
     attribute_hidden;
extern int __nscd_getservbyport_r (int port, const char *proto,
				   struct servent *result_buf, char *buf,
				   size_t buflen, struct servent **result)
     attribute_hidden;
extern int __nscd_innetgr (const char *netgroup, const char *host,
			   const char *user, const char *domain)
     attribute_hidden;
extern int __nscd_setnetgrent (const char *group, struct __netgrent *datap)
     attribute_hidden;


#endif /* _NSCD_PROTO_H */
