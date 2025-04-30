/* Entering namespaces for test case isolation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_NAMESPACE_H
#define SUPPORT_NAMESPACE_H

#include <stdbool.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Attempts to become root (or acquire root-like privileges), possibly
   with the help of user namespaces.  Return true if (restricted) root
   privileges could be attained in some way.  Print diagnostics to
   standard output.

   Note that this function generally has to be called before a process
   becomes multi-threaded, otherwise it may fail with insufficient
   privileges on systems which would support this operation for
   single-threaded processes.  */
bool support_become_root (void);

/* Return true if this process can perform a chroot operation.  In
   general, this is only possible if support_become_root has been
   called.  Note that the actual test is performed in a subprocess,
   after fork, so that the file system root of the original process is
   not changed.  */
bool support_can_chroot (void);

/* Enter a network namespace (and a UTS namespace if possible) and
   configure the loopback interface.  Return true if a network
   namespace could be created.  Print diagnostics to standard output.
   If a network namespace could be created, but networking in it could
   not be configured, terminate the process.  It is recommended to
   call support_become_root before this function so that the process
   has sufficient privileges.  */
bool support_enter_network_namespace (void);

/* Enter a mount namespace and mark / as private (not shared).  If
   this function returns true, mount operations in this process will
   not affect the host system afterwards.  */
bool support_enter_mount_namespace (void);

/* Return true if support_enter_network_namespace managed to enter a
   UTS namespace.  */
bool support_in_uts_namespace (void);

/* Invoke CALLBACK (CLOSURE) in a subprocess created using fork.
   Terminate the calling process if the subprocess exits with a
   non-zero exit status.  */
void support_isolate_in_subprocess (void (*callback) (void *), void *closure);

/* Describe the setup of a chroot environment, for
   support_chroot_create below.  */
struct support_chroot_configuration
{
  /* File contents.  The files are not created if the field is
     NULL.  */
  const char *resolv_conf;      /* /etc/resolv.conf.  */
  const char *hosts;            /* /etc/hosts.  */
  const char *host_conf;        /* /etc/host.conf.  */
  const char *aliases;          /* /etc/aliases.  */
};

/* The result of the creation of a chroot.  */
struct support_chroot
{
  /* Path information.  All these paths are relative to the parent
     chroot.  */

  /* Path to the chroot directory.  */
  char *path_chroot;

  /* Paths to files in the chroot.  These are absolute and outside of
     the chroot.  */
  char *path_resolv_conf;       /* /etc/resolv.conf.  */
  char *path_hosts;             /* /etc/hosts.  */
  char *path_host_conf;         /* /etc/host.conf.  */
  char *path_aliases;           /* /etc/aliases.  */
};

/* Create a chroot environment.  The returned data should be freed
   using support_chroot_free below.  The files will be deleted when
   the process exits.  This function does not enter the chroot.  */
struct support_chroot *support_chroot_create
  (struct support_chroot_configuration);

/* Deallocate the chroot information created by
   support_chroot_create.  */
void support_chroot_free (struct support_chroot *);

__END_DECLS

#endif
