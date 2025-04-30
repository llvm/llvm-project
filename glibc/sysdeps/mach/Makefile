# Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

ifdef in-Makerules

# Look for header files in mach/ under the top-level library source directory.
# Look for generated header files where they get created.
includes += -I$(..)mach -I$(common-objpfx)mach/

ifneq (mach,$(subdir))
# Subdirectories other than mach/ might use the generated Mach headers.
# So make sure we get a chance to run in mach/ to make them before all else.

mach-objpfx = $(common-objpfx)mach/
else
mach-objpfx = $(objpfx)
endif

# These are all the generated files that <mach.h> includes.
# Actually, it's only some of them.  We omit mach_interface.h
# because it's different in Darwin and the conditional crap is
# too much trouble.  This should suffice for getting the mach/Makefile
# rules invoked when they need to be.
mach-before-compile := $(mach-objpfx)mach-shortcuts.h \
		       $(patsubst %,$(mach-objpfx)mach/mach_%.h,\
				  port host)

ifneq (mach,$(subdir))
# This patsubst generates patterns like `m%h-shortcuts.h', which are damn
# likely to match just the corresponding particular file we want.
$(patsubst mach%,m\%h%,$(mach-before-compile)): mach-before-compile # Run only if doesn't exist.
.PHONY: mach-before-compile
mach-before-compile:
	$(MAKE) -C $(..)mach subdir=mach mach-before-compile no_deps=t generating=t

before-compile += $(mach-before-compile)
endif

ifeq (crypt,$(subdir))
  LDLIBS-crypt.so += $(objdir)/mach/libmachuser.so
else ifeq (dlfcn,$(subdir))
  LDLIBS-dl.so += $(objdir)/mach/libmachuser.so
else ifeq (nis,$(subdir))
  LDLIBS-nsl.so += $(objdir)/mach/libmachuser.so
  LDLIBS-nss_compat.so += $(objdir)/mach/libmachuser.so
else ifeq (nss,$(subdir))
  LDLIBS-nss.so += $(objdir)/mach/libmachuser.so
  LDLIBS-nss_files.so += $(objdir)/mach/libmachuser.so
  LDLIBS-nss_db.so += $(objdir)/mach/libmachuser.so
  LDLIBS-nss_compat.so += $(objdir)/mach/libmachuser.so
else ifeq (hesiod,$(subdir))
  LDLIBS-nss_hesiod.so += $(objdir)/mach/libmachuser.so
else ifeq (posix,$(subdir))
  LDLIBS-tst-rfc3484 += $(objdir)/mach/libmachuser.so
  LDLIBS-tst-rfc3484-2 += $(objdir)/mach/libmachuser.so
  LDLIBS-tst-rfc3484-3 += $(objdir)/mach/libmachuser.so
else ifeq (nscd,$(subdir))
  LDLIBS-nscd += $(objdir)/mach/libmachuser.so
endif

LDLIBS-pthread.so += $(objdir)/mach/libmachuser.so

endif	# in-Makerules
