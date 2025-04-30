/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2004.

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

#include <alloca.h>
#include <sys/stat.h>

/* This file uses the getaddrinfo code but it compiles it without NSCD
   support.  We just need a few symbol renames.  */
#define __ioctl ioctl
#define __getsockname getsockname
#define __socket socket
#define __recvmsg recvmsg
#define __bind bind
#define __sendto sendto
#define __strchrnul strchrnul
#define __getline getline
#define __qsort_r qsort_r
/* nscd uses 1MB or 2MB thread stacks.  */
#define __libc_use_alloca(size) (size <= __MAX_ALLOCA_CUTOFF)
#define __getifaddrs getifaddrs
#define __freeifaddrs freeifaddrs
#undef __fstat64
#define __fstat64 fstat64
#undef __stat64
#define __stat64 stat64

/* We are nscd, so we don't want to be talking to ourselves.  */
#undef  USE_NSCD

#include <getaddrinfo.c>

/* Support code.  */
#include <check_pf.c>
#include <check_native.c>

/* Some variables normally defined in libc.  */
nss_action_list __nss_hosts_database attribute_hidden;
