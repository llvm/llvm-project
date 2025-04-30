# Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for misc portion of the library.
#

subdir	:= misc

include ../Makeconfig

headers	:= sys/uio.h bits/uio-ext.h bits/uio_lim.h \
	   sys/ioctl.h bits/ioctls.h bits/ioctl-types.h \
	   sys/ptrace.h sys/file.h sys/dir.h sys/cdefs.h \
	   ar.h a.out.h libgen.h stab.h bits/stab.def sgtty.h \
	   ttyent.h paths.h sys/reboot.h \
	   sys/mman.h sys/param.h bits/param.h \
	   fstab.h mntent.h search.h err.h error.h \
	   sys/queue.h sysexits.h syscall.h sys/syscall.h sys/swap.h \
	   sys/select.h sys/sysinfo.h \
	   regexp.h bits/select.h bits/mman.h sys/xattr.h \
	   syslog.h sys/syslog.h \
	   bits/syslog.h bits/syslog-ldbl.h bits/syslog-path.h bits/error.h \
	   bits/select2.h bits/hwcap.h sys/auxv.h \
	   sys/sysmacros.h bits/sysmacros.h bits/types/struct_iovec.h \
	   bits/err-ldbl.h bits/error-ldbl.h \
	   sys/single_threaded.h

routines := brk sbrk sstk ioctl \
	    readv writev preadv preadv64 pwritev pwritev64 \
	    preadv2 preadv64v2 pwritev2 pwritev64v2 \
	    setreuid setregid \
	    seteuid setegid \
	    getpagesize \
	    getdtsz \
	    gethostname sethostname getdomain setdomain \
	    select pselect \
	    acct chroot fsync sync fdatasync syncfs reboot \
	    gethostid sethostid \
	    revoke vhangup \
	    swapon swapoff mktemp mkstemp mkstemp64 mkdtemp \
	    mkostemp mkostemp64 mkstemps mkstemps64 mkostemps mkostemps64 \
	    ualarm usleep \
	    gtty stty \
	    ptrace \
	    fstab mntent mntent_r \
	    utimes lutimes futimes futimesat \
	    truncate ftruncate truncate64 ftruncate64 \
	    chflags fchflags \
	    insremque getttyent getusershell getpass ttyslot \
	    syslog syscall daemon \
	    mmap mmap64 munmap mprotect msync madvise mincore remap_file_pages\
	    mlock munlock mlockall munlockall \
	    efgcvt efgcvt_r qefgcvt qefgcvt_r \
	    hsearch hsearch_r tsearch lsearch \
	    err error ustat \
	    getsysstats dirname regexp \
	    getloadavg getclktck \
	    fgetxattr flistxattr fremovexattr fsetxattr getxattr \
	    listxattr lgetxattr llistxattr lremovexattr lsetxattr \
	    removexattr setxattr getauxval ifunc-impl-list makedev \
	    allocate_once fd_to_filename single_threaded unwind-link

generated += tst-error1.mtrace tst-error1-mem.out \
  tst-allocate_once.mtrace tst-allocate_once-mem.out

aux := init-misc
install-lib := libg.a
gpl2lgpl := error.c error.h

tests := tst-dirname tst-tsearch tst-fdset tst-mntent tst-hsearch \
	 tst-error1 tst-pselect tst-insremque tst-mntent2 bug-hsearch1 \
	 tst-mntent-blank-corrupt tst-mntent-blank-passno bug18240 \
	 tst-preadvwritev tst-preadvwritev64 tst-makedev tst-empty \
	 tst-preadvwritev2 tst-preadvwritev64v2 tst-warn-wide \
	 tst-ldbl-warn tst-ldbl-error tst-dbl-efgcvt tst-ldbl-efgcvt \
	 tst-mntent-autofs tst-syscalls tst-mntent-escape tst-select \
	 tst-ioctl

tests-time64 := \
  tst-select-time64 \
  tst-pselect-time64 \
  tst-ioctl-time64 \
  # tests-time64

# Tests which need libdl.
ifeq (yes,$(build-shared))
tests += tst-gethostid
endif

tests-internal := tst-atomic tst-atomic-long tst-allocate_once
tests-static := tst-empty

# Test for the internal, non-exported __fd_to_filename function.
tests-internal += tst-fd_to_filename
tests-static += tst-fd_to_filename

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)tst-error1-mem.out \
  $(objpfx)tst-allocate_once-mem.out
endif

CFLAGS-select.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-tsearch.c += $(uses-callbacks)
CFLAGS-lsearch.c += $(uses-callbacks)
CFLAGS-pselect.c += -fexceptions
CFLAGS-readv.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-writev.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-preadv.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-preadv64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pwritev.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pwritev64.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-preadv2.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-preadv64v2.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pwritev2.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pwritev64v2.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-usleep.c += -fexceptions
CFLAGS-syslog.c += -fexceptions
CFLAGS-error.c += -fexceptions
CFLAGS-getpass.c += -fexceptions
CFLAGS-mkstemp.c += -fexceptions
CFLAGS-mkstemp64.c += -fexceptions
CFLAGS-getsysstats.c += -fexceptions
CFLAGS-getusershell.c += -fexceptions
CFLAGS-err.c += -fexceptions
CFLAGS-tst-tsearch.c += $(stack-align-test-flags)
CFLAGS-msync.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-fdatasync.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-fsync.c += -fexceptions -fasynchronous-unwind-tables

# Called during static library initialization, so turn stack-protection
# off for non-shared builds.
CFLAGS-sbrk.o = $(no-stack-protector)
CFLAGS-sbrk.op = $(no-stack-protector)
CFLAGS-brk.o = $(no-stack-protector)
CFLAGS-brk.op = $(no-stack-protector)

include ../Rules

$(objpfx)libg.a: $(dep-dummy-lib); $(make-dummy-lib)

$(objpfx)tst-tsearch: $(libm)

tst-error1-ENV = MALLOC_TRACE=$(objpfx)tst-error1.mtrace \
		 LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
tst-error1-ARGS = $(objpfx)tst-error1.out
$(objpfx)tst-error1-mem.out: $(objpfx)tst-error1.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-error1.mtrace > $@; \
	$(evaluate-test)

tst-allocate_once-ENV = MALLOC_TRACE=$(objpfx)tst-allocate_once.mtrace \
			LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)tst-allocate_once-mem.out: $(objpfx)tst-allocate_once.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-allocate_once.mtrace > $@; \
	$(evaluate-test)

$(objpfx)tst-select: $(librt)
$(objpfx)tst-select-time64: $(librt)
$(objpfx)tst-pselect: $(librt)
$(objpfx)tst-pselect-time64: $(librt)
