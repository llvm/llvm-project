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
#	Makefile for stdlib routines
#
subdir	:= stdlib

include ../Makeconfig

headers	:= stdlib.h bits/stdlib.h bits/stdlib-ldbl.h bits/stdlib-float.h      \
	   monetary.h bits/monetary-ldbl.h				      \
	   inttypes.h stdint.h bits/wordsize.h bits/timesize.h		      \
	   errno.h sys/errno.h bits/errno.h bits/types/error_t.h	      \
	   ucontext.h sys/ucontext.h bits/indirect-return.h		      \
	   alloca.h fmtmsg.h						      \
	   bits/stdlib-bsearch.h sys/random.h bits/stdint-intn.h	      \
	   bits/stdint-uintn.h bits/time64.h				      \

routines	:=							      \
	atof atoi atol atoll						      \
	abort								      \
	bsearch qsort msort						      \
	getenv putenv setenv secure-getenv				      \
	exit on_exit atexit cxa_atexit cxa_finalize old_atexit		      \
	quick_exit at_quick_exit cxa_at_quick_exit cxa_thread_atexit_impl     \
	abs labs llabs							      \
	div ldiv lldiv							      \
	mblen mbstowcs mbtowc wcstombs wctomb				      \
	random random_r rand rand_r					      \
	drand48 erand48 lrand48 nrand48 mrand48 jrand48			      \
	srand48 seed48 lcong48						      \
	drand48_r erand48_r lrand48_r nrand48_r mrand48_r jrand48_r	      \
	srand48_r seed48_r lcong48_r					      \
	drand48-iter getrandom getentropy				      \
	strfromf strfromd strfroml					      \
	strtol strtoul strtoll strtoull					      \
	strtol_l strtoul_l strtoll_l strtoull_l				      \
	strtof strtod strtold						      \
	strtof_l strtod_l strtold_l					      \
	strtof_nan strtod_nan strtold_nan				      \
	system canonicalize						      \
	a64l l64a							      \
	rpmatch strfmon strfmon_l getsubopt xpg_basename fmtmsg		      \
	getcontext setcontext makecontext swapcontext
aux =	grouping groupingwc tens_in_limb

# These routines will be omitted from the libc shared object.
# Instead the static object files will be included in a special archive
# linked against when the shared library will be used.
static-only-routines = atexit at_quick_exit

test-srcs	:= tst-fmtmsg
tests		:= tst-strtol tst-strtod testmb testrand testsort testdiv   \
		   test-canon test-canon2 tst-strtoll tst-environ	    \
		   tst-xpg-basename tst-random tst-random2 tst-bsearch	    \
		   tst-limits tst-rand48 bug-strtod tst-setcontext	    \
		   tst-setcontext2 test-a64l tst-qsort testmb2              \
		   bug-strtod2 tst-atof1 tst-atof2 tst-strtod2		    \
		   tst-rand48-2 tst-makecontext tst-strtod5		    \
		   tst-qsort2 tst-makecontext2 tst-strtod6 tst-unsetenv1    \
		   tst-makecontext3 bug-getcontext bug-fmtmsg1		    \
		   tst-secure-getenv tst-strtod-overflow tst-strtod-round   \
		   tst-tininess tst-strtod-underflow tst-setcontext3	    \
		   tst-strtol-locale tst-strtod-nan-locale tst-strfmon_l    \
		   tst-quick_exit tst-thread-quick_exit tst-width	    \
		   tst-width-stdint tst-strfrom tst-strfrom-locale	    \
		   tst-getrandom tst-atexit tst-at_quick_exit 		    \
		   tst-cxa_atexit tst-on_exit test-atexit-race 		    \
		   test-at_quick_exit-race test-cxa_atexit-race		    \
		   test-cxa_atexit-race2				    \
		   test-on_exit-race test-dlclose-exit-race 		    \
		   tst-makecontext-align test-bz22786 tst-strtod-nan-sign \
		   tst-swapcontext1 tst-setcontext4 tst-setcontext5 \
		   tst-setcontext6 tst-setcontext7 tst-setcontext8 \
		   tst-setcontext9 tst-bz20544 tst-canon-bz26341 \
		   tst-realpath

tests-internal	:= tst-strtod1i tst-strtod3 tst-strtod4 tst-strtod5i \
		   tst-tls-atexit tst-tls-atexit-nodelete
tests-static	:= tst-secure-getenv
tests-container := tst-system

ifeq ($(build-hardcoded-path-in-tests),yes)
tests += tst-empty-env
endif

LDLIBS-test-atexit-race = $(shared-thread-library)
LDLIBS-test-at_quick_exit-race = $(shared-thread-library)
LDLIBS-test-cxa_atexit-race = $(shared-thread-library)
LDLIBS-test-cxa_atexit-race2 = $(shared-thread-library)
LDLIBS-test-on_exit-race = $(shared-thread-library)
LDLIBS-tst-canon-bz26341 = $(shared-thread-library)

LDLIBS-test-dlclose-exit-race = $(shared-thread-library)
LDFLAGS-test-dlclose-exit-race = $(LDFLAGS-rdynamic)
LDLIBS-test-dlclose-exit-race-helper.so = $(libsupport) $(shared-thread-library)

ifeq ($(have-cxx-thread_local),yes)
CFLAGS-tst-quick_exit.o = -std=c++11
LDLIBS-tst-quick_exit = -lstdc++
CFLAGS-tst-thread-quick_exit.o = -std=c++11
LDLIBS-tst-thread-quick_exit = -lstdc++
$(objpfx)tst-thread-quick_exit: $(shared-thread-library)
else
tests-unsupported += tst-quick_exit tst-thread-quick_exit
endif

modules-names	= tst-tls-atexit-lib test-dlclose-exit-race-helper
extra-test-objs += $(addsuffix .os, $(modules-names))

ifeq ($(build-shared),yes)
tests		+= tst-putenv
endif

# Several mpn functions from GNU MP are used by the strtod function.
mpn-routines := inlines add_n addmul_1 cmp divmod_1 divrem udiv_qrnnd \
		lshift rshift mod_1 mul mul_1 mul_n sub_n submul_1
mpn-headers = longlong.h gmp.h gmp-impl.h gmp-mparam.h asm-syntax.h

routines := $(strip $(routines) $(mpn-routines))	\
	    dbl2mpn ldbl2mpn				\
	    mpn2flt mpn2dbl mpn2ldbl
aux += fpioconst mp_clz_tab

tests-extras += tst-putenvmod
extra-test-objs += tst-putenvmod.os

generated += isomac isomac.out tst-putenvmod.so

CFLAGS-bsearch.c += $(uses-callbacks)
CFLAGS-msort.c += $(uses-callbacks)
CFLAGS-qsort.c += $(uses-callbacks)
CFLAGS-system.c += -fexceptions
CFLAGS-system.os = -fomit-frame-pointer
CFLAGS-fmtmsg.c += -fexceptions

CFLAGS-strfmon.c += $(libio-mtsafe)
CFLAGS-strfmon_l.c += $(libio-mtsafe)

# The strfrom class of functions call __printf_fp in order to convert the
# floating-point value to characters.  This requires the value of IO_MTSAFE_IO.
CFLAGS-strfromd.c += $(libio-mtsafe)
CFLAGS-strfromf.c += $(libio-mtsafe)
CFLAGS-strfroml.c += $(libio-mtsafe)

CFLAGS-tst-bsearch.c += $(stack-align-test-flags)
CFLAGS-tst-qsort.c += $(stack-align-test-flags)
CFLAGS-tst-makecontext.c += -funwind-tables
CFLAGS-tst-makecontext2.c += $(stack-align-test-flags)

# Run a test on the header files we use.
tests-special += $(objpfx)isomac.out

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)tst-fmtmsg.out
endif

include ../Rules

ifeq ($(run-built-tests),yes)
LOCALES := cs_CZ.UTF-8 de_DE.UTF-8 en_US.ISO-8859-1 tr_TR.UTF-8 \
	   tr_TR.ISO-8859-9 tg_TJ.UTF-8 hr_HR.UTF-8 hi_IN.UTF-8 \
	   el_GR.UTF-8
include ../gen-locales.mk

$(objpfx)bug-strtod2.out: $(gen-locales)
$(objpfx)testmb2.out: $(gen-locales)
$(objpfx)tst-strtod.out: $(gen-locales)
$(objpfx)tst-strtod1i.out: $(gen-locales)
$(objpfx)tst-strtod3.out: $(gen-locales)
$(objpfx)tst-strtod4.out: $(gen-locales)
$(objpfx)tst-strtod5.out: $(gen-locales)
$(objpfx)tst-strtod5i.out: $(gen-locales)
$(objpfx)tst-strtol-locale.out: $(gen-locales)
$(objpfx)tst-strtod-nan-locale.out: $(gen-locales)
$(objpfx)tst-strfmon_l.out: $(gen-locales)
$(objpfx)tst-strfrom.out: $(gen-locales)
$(objpfx)tst-strfrom-locale.out: $(gen-locales)
$(objpfx)test-dlclose-exit-race.out: $(objpfx)test-dlclose-exit-race-helper.so
endif

# Testdir has to be named stdlib and needs to be writable
test-canon-ARGS = --test-dir=${common-objpfx}stdlib

bug-fmtmsg1-ENV = SEV_LEVEL=foo,11,newsev

$(objpfx)isomac.out: $(objpfx)isomac
	$(dir $<)$(notdir $<) '$(CC)'  \
	'-I../include $(+sysdep-includes) $(sysincludes) -I..' > $@; \
	$(evaluate-test)

isomac-CFLAGS = -O
$(objpfx)isomac: isomac.c
	$(native-compile)

$(objpfx)tst-fmtmsg.out: tst-fmtmsg.sh $(objpfx)tst-fmtmsg
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix-before-env)' \
		 '$(run-program-env)' '$(test-program-prefix-after-env)' \
		 $(common-objpfx)stdlib/; \
	$(evaluate-test)

$(objpfx)tst-putenv: $(objpfx)tst-putenvmod.so
LDFLAGS-tst-putenv = -Wl,--no-as-needed

$(objpfx)tst-putenvmod.so: $(objpfx)tst-putenvmod.os $(link-libc-deps)
	$(build-module)
libof-tst-putenvmod = extramodules

$(objpfx)bug-getcontext: $(libm)
$(objpfx)bug-strtod2: $(libm)
$(objpfx)tst-strtod-round: $(libm)
$(objpfx)tst-tininess: $(libm)
$(objpfx)tst-strtod-underflow: $(libm)
$(objpfx)tst-strtod6: $(libm)
$(objpfx)tst-strtod-nan-locale: $(libm)
$(objpfx)tst-strtod-nan-sign: $(libm)

tst-tls-atexit-lib.so-no-z-defs = yes
test-dlclose-exit-race-helper.so-no-z-defs = yes

$(objpfx)tst-tls-atexit: $(shared-thread-library)
$(objpfx)tst-tls-atexit.out: $(objpfx)tst-tls-atexit-lib.so

$(objpfx)tst-tls-atexit-nodelete: $(shared-thread-library)
$(objpfx)tst-tls-atexit-nodelete.out: $(objpfx)tst-tls-atexit-lib.so

$(objpfx)tst-setcontext3.out: tst-setcontext3.sh $(objpfx)tst-setcontext3
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix-before-env)' \
		 '$(run-program-env)' '$(test-program-prefix-after-env)' \
		 $(common-objpfx)stdlib/; \
	$(evaluate-test)
