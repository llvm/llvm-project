# Makefile for csu code for GNU C library.
# Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

# This directory contains the C startup code (that which calls main).  This
# consists of the startfile, built from start.c and installed as crt0.o
# (traditionally) or crt1.o (for ELF).	In ELF we also install crti.o and
# crtn.o, special "initializer" and "finalizer" files used in the link
# to make the .init and .fini sections work right.

subdir := csu

include ../Makeconfig

routines = init-first libc-start $(libc-init) sysdep version check_fds \
	   libc-tls dso_handle
aux	 = errno
elide-routines.os = libc-tls
csu-dummies = $(filter-out $(start-installed-name),crt1.o Mcrt1.o)
extra-objs = start.o \
	     $(start-installed-name) g$(start-installed-name) $(csu-dummies) \
	     S$(start-installed-name)
omit-deps = $(patsubst %.o,%,$(start-installed-name) g$(start-installed-name) \
			     b$(start-installed-name) $(csu-dummies) \
			     S$(start-installed-name) \
			     r$(start-installed-name) \
			     gr$(start-installed-name))
install-lib = $(start-installed-name) g$(start-installed-name) $(csu-dummies)

# No tests are allowed in the csu/ subdirectory because the startup
# code is compiled with special flags.
tests =

# static-reloc.os is a dummy object not actually used for anything.  It is
# linked into crt1.o nevertheless, which in turn is statically linked into
# applications, so that build flags matter.
# See <https://sourceware.org/ml/libc-alpha/2018-07/msg00101.html>.
#
# libc-start.os is safe to be built with stack protector since
# __libc_start_main is called after stack canary setup is done.
ssp-safe.os = static-reloc libc-start

CFLAGS-.o += $(call elide-stack-protector,.o,$(routines))
CFLAGS-.op += $(call elide-stack-protector,.op,$(routines))
CFLAGS-.oS += $(call elide-stack-protector,.oS,$(routines))
CFLAGS-.os += $(call elide-stack-protector,.os,$(filter-out \
						 $(ssp-safe.os),$(routines)))

ifeq (yes,$(build-shared))
extra-objs += S$(start-installed-name) gmon-start.os
ifneq ($(start-installed-name),$(static-start-installed-name))
extra-objs += gmon-start.o
endif
install-lib += S$(start-installed-name)
generated += start.os
else
extra-objs += gmon-start.o
endif

ifneq ($(start-installed-name),$(static-start-installed-name))
# FIXME: Only Hurd defines static-start-installed-name.  Hurd needs to
# provide special rules to support static PIE.
extra-objs += $(static-start-installed-name) g$(static-start-installed-name)
omit-deps += $(patsubst %.o,%,$(static-start-installed-name) \
			     g$(static-start-installed-name))
install-lib += $(static-start-installed-name) g$(static-start-installed-name)
else
ifeq (yes,$(enable-static-pie))
extra-objs += r$(start-installed-name) gr$(start-installed-name)
install-lib += r$(start-installed-name) gr$(start-installed-name)
endif
endif

before-compile += $(objpfx)abi-tag.h
generated += abi-tag.h

# These are the special initializer/finalizer files.  They are always the
# first and last file in the link.  crti.o ... crtn.o define the global
# "functions" _init and _fini to run the .init and .fini sections.
crtstuff = crti crtn

install-lib += $(crtstuff:=.o)
extra-objs += $(crtstuff:=.o)

ifneq ($(multidir),.)
multilib-extra-objs = $(addprefix $(multidir)/, $(install-lib))
extra-objs += $(multilib-extra-objs)
endif

extra-objs += abi-note.o init.o static-reloc.o
ifeq (yes,$(build-shared))
extra-objs += static-reloc.os
endif
asm-CPPFLAGS += -I$(objpfx).

# Enable unwinding so backtrace unwinds to __libc_start_main
CFLAGS-libc-start.c += -funwind-tables

include ../Rules

# Make these in the lib pass so they're available in time to link things with.
subdir_lib: $(extra-objs:%=$(objpfx)%)

define link-relocatable
$(CC) -nostdlib -nostartfiles -r -o $@ $^
endef

ifndef start-installed-name-rule
# We link the ELF startfile along with a SHT_NOTE section indicating
# the kernel ABI the binaries linked with this library will require.
$(objpfx)$(start-installed-name): $(objpfx)start.o $(objpfx)abi-note.o \
				  $(objpfx)init.o $(objpfx)static-reloc.o
	$(link-relocatable)
$(objpfx)r$(start-installed-name): $(objpfx)start.o $(objpfx)abi-note.o \
				  $(objpfx)init.o
	$(link-relocatable)
$(objpfx)S$(start-installed-name): $(objpfx)start.os $(objpfx)abi-note.o \
				  $(objpfx)init.o
	$(link-relocatable)
endif

# The profiling startfile is made by linking together the normal
# startfile with gmon-start.o, which defines a constructor function
# to turn on profiling code at startup.
ifeq (yes,$(build-shared))
$(objpfx)g$(start-installed-name): \
  $(objpfx)g%: $(objpfx)S% $(objpfx)gmon-start.os $(objpfx)static-reloc.os
	$(link-relocatable)
$(objpfx)gr$(start-installed-name): \
  $(objpfx)gr%: $(objpfx)r% $(objpfx)gmon-start.o
	$(link-relocatable)
ifneq ($(start-installed-name),$(static-start-installed-name))
$(objpfx)g$(static-start-installed-name): \
  $(objpfx)g%: $(objpfx)% $(objpfx)gmon-start.o
	$(link-relocatable)
endif
else
$(addprefix $(objpfx),$(sort g$(start-installed-name) \
			     g$(static-start-installed-name))): \
  $(objpfx)g%: $(objpfx)% $(objpfx)gmon-start.o
	$(link-relocatable)
endif

# These extra files are sometimes expected by system standard linking
# procedures, but we have nothing for them to do.  So compile empty files.
$(addprefix $(objpfx),$(filter-out $(start-installed-name), $(csu-dummies))):\
  $(before-compile)
	$(COMPILE.c) -o $@ -x c /dev/null

# These headers are used by the startup code.
$(objpfx)abi-tag.h: $(..)abi-tags
	$(make-target-directory)
	rm -f $@.new
	sed -e 's/#.*$$//' -e '/^[	]*$$/d' $< | \
	while read conf tagos tagver; do \
	  test `expr '$(config-machine)-$(config-vendor)-$(config-os)' \
		     : "$$conf"` != 0 || continue; \
	  ( echo "$$tagos" | \
	    sed -e 's/[^0-9xXa-fA-F 	]//' \
		-e 's/^/#define __ABI_TAG_OS /'; \
	    echo "#ifndef __ABI_TAG_VERSION"; \
	    echo "$$tagver" | \
	    sed -e 's/[^0-9xXa-fA-F]/ /g' -e 's/ *$$//' \
	      -e 's/ /,/g' -e 's/^/# define __ABI_TAG_VERSION /'; \
	    echo "#endif" ) > $@.new; \
	done
	if test -r $@.new; then mv -f $@.new $@; \
	else echo >&2 'This configuration not matched in $<'; exit 1; fi

ifneq ($(multidir),.)
$(addprefix $(objpfx)$(multidir)/, $(install-lib)): $(addprefix $(objpfx), $(install-lib))
	$(make-link-multidir)
endif
