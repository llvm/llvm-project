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
#	Specific makefile for stdio-common.
#
subdir	:= stdio-common

include ../Makeconfig

headers	:= stdio_ext.h printf.h bits/printf-ldbl.h bits/stdio_lim.h

routines	:=							      \
	ctermid cuserid							      \
	_itoa _itowa itoa-digits itoa-udigits itowa-digits		      \
	vfprintf vprintf printf_fp reg-printf printf-prs printf_fphex	      \
	reg-modifier reg-type						      \
	printf_size fprintf printf snprintf sprintf asprintf dprintf	      \
	vfwprintf vfscanf vfwscanf					      \
	fscanf scanf sscanf						      \
	perror psignal							      \
	tmpfile tmpfile64 tmpnam tmpnam_r tempnam tempname		      \
	getline getw putw						      \
	remove rename renameat renameat2				      \
	flockfile ftrylockfile funlockfile				      \
	isoc99_scanf isoc99_vscanf isoc99_fscanf isoc99_vfscanf isoc99_sscanf \
	isoc99_vsscanf							      \
	psiginfo gentempfd						      \
	vfscanf-internal vfwscanf-internal iovfscanf			      \
	vfprintf-internal vfwprintf-internal

aux	:= errlist siglist printf-parsemb printf-parsewc fxprintf

tests := tstscanf test_rdwr test-popen tstgetln test-fseek \
	 temptest tst-fileno test-fwrite tst-ungetc tst-ferror \
	 xbug errnobug \
	 bug1 bug2 bug3 bug4 bug5 bug6 bug7 bug8 bug9 bug10 bug11 bug12 bug13 \
	 tfformat tiformat tllformat tstdiomisc tst-printfsz tst-wc-printf \
	 scanf1 scanf2 scanf3 scanf4 scanf5 scanf7 scanf8 scanf9 scanf10 \
	 scanf11 scanf12 tst-tmpnam tst-cookie tst-obprintf tst-sscanf \
	 tst-swprintf tst-fseek tst-fmemopen test-vfprintf tst-gets \
	 tst-perror tst-sprintf tst-rndseek tst-fdopen tst-fphex bug14 \
	 tst-popen tst-unlockedio tst-fmemopen2 tst-put-error tst-fgets \
	 tst-fwrite bug16 bug17 tst-swscanf tst-sprintf2 bug18 bug18a \
	 bug19 bug19a tst-popen2 scanf13 scanf14 scanf15 bug20 bug21 bug22 \
	 scanf16 scanf17 tst-setvbuf1 tst-grouping bug23 bug24 \
	 bug-vfprintf-nargs tst-long-dbl-fphex tst-fphex-wide tst-sprintf3 \
	 bug25 tst-printf-round bug23-2 bug23-3 bug23-4 bug26 tst-fmemopen3 \
	 tst-printf-bz18872 tst-vfprintf-width-prec tst-fmemopen4 \
	 tst-vfprintf-user-type \
	 tst-vfprintf-mbs-prec \
	 tst-scanf-round \
	 tst-renameat2 tst-bz11319 tst-bz11319-fortify2 \
	 scanf14a scanf16a \
	 tst-printf-bz25691 \
	 tst-vfprintf-width-prec-alloc \
	 tst-printf-fp-free \
	 tst-printf-fp-leak \
	 test-strerr


test-srcs = tst-unbputc tst-printf tst-printfsz-islongdouble

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)tst-unbputc.out $(objpfx)tst-printf.out \
		 $(objpfx)tst-printf-bz18872-mem.out \
		 $(objpfx)tst-setvbuf1-cmp.out \
		 $(objpfx)tst-vfprintf-width-prec-mem.out \
		 $(objpfx)tst-printfsz-islongdouble.out \
		 $(objpfx)tst-printf-bz25691-mem.out \
		 $(objpfx)tst-printf-fp-free-mem.out \
		 $(objpfx)tst-printf-fp-leak-mem.out
generated += tst-printf-bz18872.c tst-printf-bz18872.mtrace \
	     tst-printf-bz18872-mem.out \
	     tst-vfprintf-width-prec.mtrace tst-vfprintf-width-prec-mem.out \
	     tst-printf-bz25691.mtrace tst-printf-bz25691-mem.out \
	     tst-printf-fp-free.mtrace tst-printf-fp-free-mem.out \
	     tst-printf-fp-leak.mtrace tst-printf-fp-leak-mem.out
endif

tests-special += $(objpfx)tst-errno-manual.out

include ../Rules

ifeq ($(run-built-tests),yes)
LOCALES := de_DE.ISO-8859-1 de_DE.UTF-8 en_US.ISO-8859-1 ja_JP.EUC-JP
include ../gen-locales.mk

$(objpfx)bug14.out: $(gen-locales)
$(objpfx)scanf13.out: $(gen-locales)
$(objpfx)test-vfprintf.out: $(gen-locales)
$(objpfx)tst-grouping.out: $(gen-locales)
$(objpfx)tst-sprintf.out: $(gen-locales)
$(objpfx)tst-sscanf.out: $(gen-locales)
$(objpfx)tst-swprintf.out: $(gen-locales)
$(objpfx)tst-vfprintf-mbs-prec.out: $(gen-locales)
endif

tst-printf-bz18872-ENV = MALLOC_TRACE=$(objpfx)tst-printf-bz18872.mtrace \
			LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
tst-vfprintf-width-prec-ENV = \
  MALLOC_TRACE=$(objpfx)tst-vfprintf-width-prec.mtrace \
  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
tst-printf-bz25691-ENV = \
  MALLOC_TRACE=$(objpfx)tst-printf-bz25691.mtrace \
  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
tst-printf-fp-free-ENV = \
  MALLOC_TRACE=$(objpfx)tst-printf-fp-free.mtrace \
  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
tst-printf-fp-leak-ENV = \
  MALLOC_TRACE=$(objpfx)tst-printf-fp-leak.mtrace \
  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so

$(objpfx)tst-unbputc.out: tst-unbputc.sh $(objpfx)tst-unbputc
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix)'; \
	$(evaluate-test)

$(objpfx)tst-printf.out: tst-printf.sh $(objpfx)tst-printf
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix)'; \
	$(evaluate-test)

$(objpfx)tst-printfsz-islongdouble.out: \
  tst-printfsz-islongdouble.sh $(objpfx)tst-printfsz-islongdouble
	$(SHELL) $^ '$(test-program-prefix)' $@; \
	$(evaluate-test)

# We generate this source because it requires a printf invocation with
# 10K arguments.
$(objpfx)tst-printf-bz18872.c: tst-printf-bz18872.sh
	rm -f $@ && $(BASH) $^ > $@.new && mv $@.new $@

$(objpfx)tst-%-mem.out: $(objpfx)tst-%.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-$*.mtrace > $@; \
	$(evaluate-test)

errlist-h = $(firstword $(wildcard $(addsuffix /errlist.h,$(sysdirs) .)))

$(objpfx)tst-errno-manual.out: tst-errno-manual.py \
			       $(errlist-h) \
			       $(..)manual/errno.texi
	$(PYTHON) tst-errno-manual.py -m $(..)manual/errno.texi \
				      -e $(errlist-h) > $@; \
	$(evaluate-test)

CFLAGS-vfprintf.c += -Wno-uninitialized
CFLAGS-vfwprintf.c += -Wno-uninitialized

CFLAGS-tmpfile.c += -fexceptions
CFLAGS-tmpfile64.c += -fexceptions
CFLAGS-tempname.c += -fexceptions
CFLAGS-psignal.c += -fexceptions
CFLAGS-vprintf.c += -fexceptions
CFLAGS-cuserid.c += -fexceptions

CFLAGS-vfprintf.c += -fexceptions
CFLAGS-fprintf.c += -fexceptions
CFLAGS-printf.c += -fexceptions
CFLAGS-vfwprintf.c += -fexceptions
CFLAGS-vfscanf.c += -fexceptions
CFLAGS-vfwscanf.c += -fexceptions
CFLAGS-fscanf.c += -fexceptions
CFLAGS-scanf.c += -fexceptions
CFLAGS-isoc99_vfscanf.c += -fexceptions
CFLAGS-isoc99_vscanf.c += -fexceptions
CFLAGS-isoc99_fscanf.c += -fexceptions
CFLAGS-isoc99_scanf.c += -fexceptions
CFLAGS-errlist.c += $(fno-unit-at-a-time)
CFLAGS-siglist.c += $(fno-unit-at-a-time)

# scanf14a.c and scanf16a.c test a deprecated extension which is no
# longer visible under most conformance levels; see the source files
# for more detail.
CFLAGS-scanf14a.c += -std=gnu89
CFLAGS-scanf16a.c += -std=gnu89

CFLAGS-bug3.c += -DOBJPFX=\"$(objpfx)\"
CFLAGS-bug4.c += -DOBJPFX=\"$(objpfx)\"
CFLAGS-bug5.c += -DOBJPFX=\"$(objpfx)\"
CFLAGS-test-fseek.c += -DOBJPFX=\"$(objpfx)\"
CFLAGS-test-popen.c += -DOBJPFX=\"$(objpfx)\"
CFLAGS-test_rdwr.c += -DOBJPFX=\"$(objpfx)\"

# tst-gets.c tests a deprecated function.
CFLAGS-tst-gets.c += -Wno-deprecated-declarations

# BZ #11319 was first fixed for regular vdprintf, then reopened because
# the fortified version had the same bug.
CFLAGS-tst-bz11319-fortify2.c += -D_FORTIFY_SOURCE=2

CPPFLAGS += $(libio-mtsafe)

$(objpfx)tst-setvbuf1.out: /dev/null $(objpfx)tst-setvbuf1
	$(test-program-cmd) > $@ 2>&1; \
	$(evaluate-test)

$(objpfx)tst-setvbuf1-cmp.out: tst-setvbuf1.expect $(objpfx)tst-setvbuf1.out
	cmp $^ > $@; \
	$(evaluate-test)

$(objpfx)tst-printf-round: $(libm)
$(objpfx)tst-scanf-round: $(libm)
