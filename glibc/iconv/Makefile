# Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#	Makefile for iconv.
#
subdir	:= iconv

include ../Makeconfig

headers		= iconv.h gconv.h
routines	= iconv_open iconv iconv_close \
		  gconv_open gconv gconv_close gconv_db gconv_conf \
		  gconv_builtin gconv_simple gconv_trans gconv_cache
routines	+= gconv_dl gconv_charset

vpath %.c ../locale/programs ../intl

iconv_prog-modules = iconv_charmap charmap charmap-dir linereader \
		     dummy-repertoire simple-hash xstrdup xmalloc \
		     record-status
iconvconfig-modules = strtab xmalloc xasprintf xstrdup hash-string
extra-objs	   = $(iconv_prog-modules:=.o) $(iconvconfig-modules:=.o)
CFLAGS-iconv_prog.c += -I../locale/programs
CFLAGS-iconv_charmap.c += -I../locale/programs
CFLAGS-dummy-repertoire.c += -I../locale/programs
CFLAGS-charmap.c += -DCHARMAP_PATH='"$(i18ndir)/charmaps"' \
		   -DDEFAULT_CHARMAP=null_pointer -DNEED_NULL_POINTER
CFLAGS-linereader.c += -DNO_TRANSLITERATION
CFLAGS-simple-hash.c += -I../locale

tests	= tst-iconv1 tst-iconv2 tst-iconv3 tst-iconv4 tst-iconv5 tst-iconv6 \
	  tst-iconv7 tst-iconv8 tst-iconv-mt tst-iconv-opt

others		= iconv_prog iconvconfig
install-others-programs	= $(inst_bindir)/iconv
install-sbin	= iconvconfig

CFLAGS-gconv_cache.c += -DGCONV_DIR='"$(gconvdir)"'
CFLAGS-gconv_conf.c += -DGCONV_PATH='"$(gconvdir)"'
CFLAGS-iconvconfig.c += -DGCONV_PATH='"$(gconvdir)"' -DGCONV_DIR='"$(gconvdir)"'

# Set libof-* for each routine.
cpp-srcs-left := $(iconv_prog-modules) $(iconvconfig-modules)
lib := iconvprogs
include $(patsubst %,$(..)libof-iterator.mk,$(cpp-srcs-left))

ifeq ($(run-built-tests),yes)
xtests-special += $(objpfx)test-iconvconfig.out
tests-special += $(objpfx)tst-iconv_prog.out
endif

# Make a copy of the file because gconv module names are constructed
# relative to the path of the configuration file.
$(objpfx)gconv-modules: test-gconv-modules
	cp $< $@

$(objpfx)tst-iconv-mt: $(shared-thread-library)

ifeq (yes,$(build-shared))
tests += tst-gconv-init-failure
modules-names += tst-gconv-init-failure-mod
modules-names-tests += tst-gconv-init-failure-mod
$(objpfx)tst-gconv-init-failure-mod.so: $(libsupport)
$(objpfx)tst-gconv-init-failure.out: \
 $(objpfx)gconv-modules $(objpfx)tst-gconv-init-failure-mod.so
endif

include ../Rules

ifeq ($(run-built-tests),yes)
LOCALES := en_US.UTF-8
include ../gen-locales.mk

$(objpfx)tst-iconv-opt.out: $(gen-locales)
endif

$(inst_bindir)/iconv: $(objpfx)iconv_prog $(+force)
	$(do-install-program)

$(objpfx)iconv_prog: $(iconv_prog-modules:%=$(objpfx)%.o)
$(objpfx)iconvconfig: $(iconvconfig-modules:%=$(objpfx)%.o)

$(objpfx)test-iconvconfig.out: $(objpfx)iconvconfig
	(set -e; \
	 tmp=$(objpfx)gconv-modules.cache.$$$$; \
	 rm -f $$tmp; \
	 $(run-program-prefix) $(objpfx)iconvconfig \
		--output=$$tmp --nostdlib $(inst_gconvdir); \
	 cmp $$tmp $(inst_gconvdir)/gconv-modules.cache; \
	 rm -f $$tmp) > $@; \
	$(evaluate-test)

$(objpfx)tst-iconv_prog.out: tst-iconv_prog.sh $(objpfx)iconv_prog
	$(BASH) $< $(common-objdir) '$(test-wrapper-env)' \
		 '$(run-program-env)' > $@; \
	$(evaluate-test)
