/* Header for nscd SELinux access controls.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Matthew Rickard <mjricka@epoch.ncsc.mil>, 2004.

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

#ifndef _SELINUX_H
#define _SELINUX_H 1

#include "nscd.h"
#ifdef HAVE_LIBCAP
# include <sys/capability.h>
#endif

#ifdef HAVE_SELINUX
/* Global variable to tell if the kernel has SELinux support.  */
extern int selinux_enabled;

/* Define this for AVC stat usage.  */
struct avc_cache_stats;

/* Initialize the userspace AVC.  */
extern void nscd_avc_init (void);
/* Determine if we are running on an SELinux kernel.  */
extern void nscd_selinux_enabled (int *selinux_enabled);
/* Check if the client has permission for the request type.  */
extern int nscd_request_avc_has_perm (int fd, request_type req);
/* Initialize AVC statistic information.  */
extern void nscd_avc_cache_stats (struct avc_cache_stats *cstats);
/* Display statistics on AVC usage.  */
extern void nscd_avc_print_stats (struct avc_cache_stats *cstats);

# ifdef HAVE_LIBCAP
/* Preserve capabilities to connect to the audit daemon. */
extern cap_t preserve_capabilities (void);
/* Install final capabilities.  */
extern void install_real_capabilities (cap_t new_caps);
# endif
#else
# define selinux_enabled 0
# define nscd_avc_init() (void) 0
# define nscd_selinux_enabled(selinux_enabled) (void) 0
# define nscd_request_avc_has_perm(fd, req) 0
# define nscd_avc_cache_stats(cstats) (void) 0
# define nscd_avc_print_stats(cstats) (void) 0
#endif /* HAVE_SELINUX */

#endif /* _SELINUX_H */
