# Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#
#	Sub-makefile for I/O portion of the library.
#
subdir	:= io

include ../Makeconfig

headers := sys/stat.h bits/stat.h sys/statfs.h bits/statfs.h sys/vfs.h \
	   sys/statvfs.h bits/statvfs.h fcntl.h sys/fcntl.h bits/fcntl.h \
	   poll.h sys/poll.h bits/poll.h bits/fcntl2.h bits/poll2.h \
	   bits/statx.h bits/statx-generic.h bits/types/struct_statx.h \
	   bits/types/struct_statx_timestamp.h \
	   utime.h ftw.h fts.h sys/sendfile.h

routines :=								\
	utime								\
	mkfifo mkfifoat							\
	stat fstat lstat stat64 fstat64 lstat64 fstatat fstatat64	\
	statx								\
	mknod mknodat 							\
	statfs fstatfs statfs64 fstatfs64				\
	statvfs fstatvfs statvfs64 fstatvfs64				\
	umask chmod fchmod lchmod fchmodat				\
	mkdir mkdirat							\
	open open_2 open64 open64_2 openat openat_2 openat64 openat64_2	\
	read write lseek lseek64 access euidaccess faccessat		\
	fcntl fcntl64 flock lockf lockf64				\
	close dup dup2 dup3 pipe pipe2					\
	creat creat64							\
	chdir fchdir							\
	getcwd getwd getdirname						\
	chown fchown lchown fchownat					\
	ttyname ttyname_r isatty					\
	link linkat symlink symlinkat readlink readlinkat		\
	unlink unlinkat rmdir						\
	ftw ftw64 fts fts64 poll ppoll					\
	posix_fadvise posix_fadvise64					\
	posix_fallocate posix_fallocate64				\
	sendfile sendfile64 copy_file_range 				\
	utimensat futimens file_change_detection			\
	fts64-time64							\
	ftw64-time64							\
	closefrom

others		:= pwd
test-srcs	:= ftwtest ftwtest-time64
tests		:= test-utime test-stat test-stat2 test-lfs tst-getcwd \
		   tst-fcntl bug-ftw1 bug-ftw2 bug-ftw3 bug-ftw4 tst-statvfs \
		   tst-openat tst-unlinkat tst-fstatat tst-futimesat \
		   tst-renameat tst-fchownat tst-fchmodat tst-faccessat \
		   tst-symlinkat tst-linkat tst-readlinkat tst-mkdirat \
		   tst-mknodat tst-mkfifoat tst-ttyname_r bug-ftw5 \
		   tst-posix_fallocate tst-posix_fallocate64 \
		   tst-fts tst-fts-lfs tst-open-tmpfile \
		   tst-copy_file_range tst-getcwd-abspath tst-lockf \
		   tst-ftw-lnk tst-lchmod \
		   tst-ftw-bz26353 tst-stat tst-stat-lfs \
		   tst-utime \
		   tst-utimes \
		   tst-futimes \
		   tst-lutimes \
		   tst-futimens \
		   tst-utimensat \
		   tst-closefrom \

tests-time64 := \
  tst-futimens-time64 \
  tst-futimes-time64\
  tst-fts-time64 \
  tst-lutimes-time64 \
  tst-stat-time64 \
  tst-futimesat-time64 \
  tst-utime-time64 \
  tst-utimensat-time64 \
  tst-utimes-time64 \
  tst-fcntl-time64 \
  # tests-time64

# Likewise for statx, but we do not need static linking here.
tests-internal += tst-statx tst-file_change_detection
tests-static += tst-statx

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)ftwtest.out \
		 $(objpfx)ftwtest-time64.out
endif

include ../Rules

CFLAGS-open.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-open64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-creat.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-creat64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-fcntl.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-fcntl64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-poll.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-ppoll.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-lockf.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-lockf64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-statfs.c += -fexceptions
CFLAGS-fstatfs.c += -fexceptions
CFLAGS-statvfs.c += -fexceptions
CFLAGS-fstatvfs.c += -fexceptions
CFLAGS-fts.c += -Wno-uninitialized $(uses-callbacks) -fexceptions
CFLAGS-fts64.c += -Wno-uninitialized $(uses-callbacks) -fexceptions
CFLAGS-fts64-time64.c += -Wno-uninitialized $(uses-callbacks) -fexceptions
CFLAGS-ftw.c += $(uses-callbacks) -fexceptions
CFLAGS-ftw64.c += $(uses-callbacks) -fexceptions
CFLAGS-ftw64-time64.c += $(uses-callbacks) -fexceptions
CFLAGS-posix_fallocate.c += -fexceptions
CFLAGS-posix_fallocate64.c += -fexceptions
CFLAGS-fallocate.c += -fexceptions
CFLAGS-fallocate64.c += -fexceptions
CFLAGS-read.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-write.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-close.c += -fexceptions -fasynchronous-unwind-tables

CFLAGS-test-stat.c += -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE
CFLAGS-test-lfs.c += -D_LARGEFILE64_SOURCE

test-stat2-ARGS = Makefile . $(objpfx)test-stat2

tst-statvfs-ARGS = $(objpfx)tst-statvfs tst-statvfs.c /tmp

tst-open-tmpfile-ARGS = --test-dir=$(objpfx)

CFLAGS-ftwtest-time64.c += -D_TIME_BITS=64 -D_FILE_OFFSET_BITS=64

ifeq ($(run-built-tests),yes)
$(objpfx)ftwtest.out: ftwtest-sh $(objpfx)ftwtest
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd)' > $@; \
	$(evaluate-test)

$(objpfx)ftwtest-time64.out: ftwtest-sh $(objpfx)ftwtest-time64
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd)' > $@; \
	$(evaluate-test)
endif
