# Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for login portion of the library.
#

subdir	:= login

include ../Makeconfig

headers	:= utmp.h bits/utmp.h lastlog.h pty.h

routines := getlogin getlogin_r setlogin getlogin_r_chk \
	    getutent getutent_r getutid getutline getutid_r getutline_r \
	    utmp_file utmpname updwtmp getpt grantpt unlockpt ptsname \
	    ptsname_r_chk login login_tty logout logwtmp openpty forkpty

CFLAGS-grantpt.c += -DLIBEXECDIR='"$(libexecdir)"'

others = utmpdump

ifeq (yes,$(build-pt-chown))
others += pt_chown
others-pie = pt_chown
install-others-programs = $(inst_libexecdir)/pt_chown
endif

subdir-dirs = programs
vpath %.c programs

tests := tst-utmp tst-utmpx tst-grantpt tst-ptsname tst-getlogin tst-updwtmpx \
  tst-pututxline-lockfail tst-pututxline-cache

ifeq ($(have-GLIBC_2.33),yes)
# Empty compatibility library for old binaries.
extra-libs      := libutil
extra-libs-others := $(extra-libs)
libutil-routines := libutil-compat
libutil-shared-only-routines := libutil-compat

# Pretend that libutil.so is a linker script, so that the symbolic
# link is not installed.
install-lib-ldscripts = libutil.so
$(inst_libdir)/libutil.so:
endif # $(have-GLIBC_2.33)

include ../Rules

CFLAGS-getpt.c += -fexceptions

ifeq (yesyes,$(have-fpie)$(build-shared))
pt_chown-cflags += $(pie-ccflag)
endif
ifeq (yes,$(have-libcap))
libcap = -lcap
endif
CFLAGS-pt_chown.c += $(pt_chown-cflags)
LDLIBS-pt_chown = $(libcap)
ifeq (yesyes,$(have-fpie)$(build-shared))
LDFLAGS-pt_chown = -Wl,-z,now
endif

# pt_chown needs to be setuid root.
$(inst_libexecdir)/pt_chown: $(objpfx)pt_chown $(+force)
	$(make-target-directory)
	-$(INSTALL_PROGRAM) -m 4755 -o root $< $@

$(objpfx)tst-pututxline-lockfail: $(shared-thread-library)
$(objpfx)tst-pututxline-cache: $(shared-thread-library)
