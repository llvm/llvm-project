#!/bin/bash
# Copyright (C) 2003-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

# This script creates a list of data types where each type is followed
# by the C++ mangled name for that type.  That list is then compared
# against the list in the c++-types.data file for the platform being
# checked.  Any difference between the two would mean that the C++ ABI
# had changed and that should not happen even if the change is compatible
# at the C language level.

#
# The list of data types has been created with
# cat <<EOF |
# #include <sys/types.h>
# #include <unistd.h>
# #include <sys/resource.h>
# #include <sys/stat.h>
# EOF
# gcc -D_GNU_SOURCE -E - |
# egrep '^typedef.*;$' |
# sed 's/^typedef[[:space:]]*//;s/\([[:space:]]\{1,\}__attribute__.*\);/;/;s/.*[[:space:]]\([*]\|\)\(.*\);/\2/' |
# egrep -v '^_' |
# LC_ALL=C sort -u
#
data=$1
shift
cxx=$(echo $* | sed 's/-fgnu89-inline//')
while read t; do
  echo -n "$t:"
  $cxx -S -xc++ -o - -D_GNU_SOURCE <(cat <<EOF
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
void foo ($t) { }
EOF
) |
  sed 's/[[:space:]]*[.]globa\?l[[:space:]]*_Z3foo\([_[:alnum:]]*\).*/\1/p;d'
done <<EOF |
blkcnt64_t
blkcnt_t
blksize_t
caddr_t
clockid_t
clock_t
daddr_t
dev_t
fd_mask
fsblkcnt64_t
fsblkcnt_t
fsfilcnt64_t
fsfilcnt_t
fsid_t
gid_t
id_t
ino64_t
ino_t
int16_t
int32_t
int64_t
int8_t
intptr_t
key_t
loff_t
mode_t
nlink_t
off64_t
off_t
pid_t
pthread_attr_t
pthread_barrier_t
pthread_barrierattr_t
pthread_cond_t
pthread_condattr_t
pthread_key_t
pthread_mutex_t
pthread_mutexattr_t
pthread_once_t
pthread_rwlock_t
pthread_rwlockattr_t
pthread_spinlock_t
pthread_t
quad_t
register_t
rlim64_t
rlim_t
sigset_t
size_t
socklen_t
ssize_t
suseconds_t
time_t
u_char
uid_t
uint
u_int
u_int16_t
u_int32_t
u_int64_t
u_int8_t
ulong
u_long
u_quad_t
useconds_t
ushort
u_short
EOF
diff -N -U0 $data -
