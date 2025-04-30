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

ifeq ($(subdir),login)
sysdep_routines += setutxent getutxent endutxent getutxid getutxline \
		   pututxline utmpxname updwtmpx getutmpx getutmp

sysdep_headers += utmpx.h bits/utmpx.h
endif


ifeq ($(subdir),inet)
sysdep_headers += netinet/udp.h netinet/ip_icmp.h
endif


ifeq ($(subdir),misc)
sysdep_headers += sys/mtio.h
endif


ifeq ($(subdir),csu)
routines += unwind-resume
shared-only-routines += unwind-resume
CFLAGS-unwind-resume.c += -fexceptions -fasynchronous-unwind-tables
endif

ifeq ($(subdir),rt)
librt-sysdep_routines += rt-unwind-resume
librt-shared-only-routines += rt-unwind-resume
CFLAGS-rt-unwind-resume.c += -fexceptions -fasynchronous-unwind-tables
endif
