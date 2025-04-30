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
#	Makefile for catgets.
#
subdir	:= catgets

include ../Makeconfig

headers		= nl_types.h
routines	= catgets open_catalog
others		= gencat
install-bin	= gencat
extra-objs	= $(gencat-modules:=.o)

tests = tst-catgets
test-srcs = test-gencat

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)de/libc.cat $(objpfx)test1.cat $(objpfx)test2.cat \
		 $(objpfx)sample.SJIS.cat $(objpfx)test-gencat.out
tests-special += $(objpfx)tst-catgets-mem.out
endif

gencat-modules	= xmalloc
others-extras   = $(gencat-modules)

# To find xmalloc.c
vpath %.c ../locale/programs


include ../Rules

$(objpfx)gencat: $(gencat-modules:%=$(objpfx)%.o)

catgets-CPPFLAGS := -DNLSPATH='"$(localedir)/%L/%N:$(localedir)/%L/LC_MESSAGES/%N:$(localedir)/%l/%N:$(localedir)/%l/LC_MESSAGES/%N:"'

generated += de.msg test1.cat test1.h test2.cat test2.h sample.SJIS.cat \
	     test-gencat.h
generated += tst-catgets.mtrace tst-catgets-mem.out

generated-dirs += de

tst-catgets-ENV = NLSPATH="$(objpfx)%l/%N.cat" LANG=de \
		  MALLOC_TRACE=$(objpfx)tst-catgets.mtrace \
		  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so

ifeq ($(run-built-tests),yes)
# This test just checks whether the program produces any error or not.
# The result is not tested.
$(objpfx)test1.cat: test1.msg $(objpfx)gencat
	$(built-program-cmd-before-env) \
	$(run-program-env) LC_ALL=hr_HR.ISO-8859-2 \
	$(built-program-cmd-after-env) -H $(objpfx)test1.h $@ $<; \
	$(evaluate-test)
$(objpfx)test2.cat: test2.msg $(objpfx)gencat
	$(built-program-cmd) -H $(objpfx)test2.h $@ $<; \
	$(evaluate-test)
$(objpfx)de/libc.cat: $(objpfx)de.msg $(objpfx)gencat
	$(make-target-directory)
	$(built-program-cmd-before-env) \
	$(run-program-env) LC_ALL=de_DE.ISO-8859-1 \
	$(built-program-cmd-after-env) $@ $<; \
	$(evaluate-test)
$(objpfx)tst-catgets.out: $(objpfx)de/libc.cat

# Generate a non-simple input file.
$(objpfx)de.msg: xopen-msg.awk $(..)po/de.po
	LC_ALL=C $(AWK) -f $^ $< > $@

$(objpfx)test-gencat.out: test-gencat.sh $(objpfx)test-gencat \
			  $(objpfx)sample.SJIS.cat
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)'; \
	$(evaluate-test)

$(objpfx)sample.SJIS.cat: sample.SJIS $(objpfx)gencat
	$(built-program-cmd) -H $(objpfx)test-gencat.h < $(word 1,$^) > $@; \
	$(evaluate-test)

$(objpfx)tst-catgets-mem.out: $(objpfx)tst-catgets.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-catgets.mtrace > $@; \
	$(evaluate-test)
endif
