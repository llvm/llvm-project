# Copyright (C) 1998-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for debug portion of the library.
#
subdir	:= debug

include ../Makeconfig

headers	:= execinfo.h

# Note that ptsname_r_chk and getlogin_r are not here, but in
# login/Makefile instead.  If that subdir is omitted from the
# build, its _FORTIFY_SOURCE support will be too.
routines  = backtrace backtracesyms backtracesymsfd noophooks \
	    memcpy_chk memmove_chk mempcpy_chk memset_chk stpcpy_chk \
	    strcat_chk strcpy_chk strncat_chk strncpy_chk stpncpy_chk \
	    sprintf_chk vsprintf_chk snprintf_chk vsnprintf_chk \
	    printf_chk fprintf_chk vprintf_chk vfprintf_chk \
	    gets_chk chk_fail readonly-area fgets_chk fgets_u_chk \
	    read_chk pread_chk pread64_chk recv_chk recvfrom_chk \
	    readlink_chk readlinkat_chk getwd_chk getcwd_chk \
	    realpath_chk fread_chk fread_u_chk \
	    wctomb_chk wcscpy_chk wmemcpy_chk wmemmove_chk wmempcpy_chk \
	    wcpcpy_chk wcsncpy_chk wcscat_chk wcsncat_chk wmemset_chk \
	    wcpncpy_chk \
	    swprintf_chk vswprintf_chk wprintf_chk fwprintf_chk \
	    vwprintf_chk vfwprintf_chk fgetws_chk fgetws_u_chk \
	    confstr_chk getgroups_chk ttyname_r_chk \
	    gethostname_chk getdomainname_chk wcrtomb_chk mbsnrtowcs_chk \
	    wcsnrtombs_chk mbsrtowcs_chk wcsrtombs_chk mbstowcs_chk \
	    wcstombs_chk asprintf_chk vasprintf_chk dprintf_chk \
	    vdprintf_chk obprintf_chk vobprintf_chk \
	    longjmp_chk ____longjmp_chk \
	    fdelt_chk poll_chk ppoll_chk \
	    explicit_bzero_chk \
	    stack_chk_fail fortify_fail \
	    $(static-only-routines)
static-only-routines := stack_chk_fail_local

# Don't add stack_chk_fail_local.o to libc.a since __stack_chk_fail_local
# is an alias of __stack_chk_fail in stack_chk_fail.o.
elide-routines.o := stack_chk_fail_local

# Building the stack-protector failure routines with stack protection
# is not required since we have already failed a stack check and are
# exiting the process.  However, the local aliases which jump to the
# real routines should still be compiled with stack protection
# (stack_chk_fail_local.c), so that the statically linked parts of the
# library have the expected flags.

CFLAGS-stack_chk_fail.c += $(no-stack-protector)

CFLAGS-backtrace.c += -fno-omit-frame-pointer -funwind-tables
CFLAGS-sprintf_chk.c += $(libio-mtsafe)
CFLAGS-snprintf_chk.c += $(libio-mtsafe)
CFLAGS-vsprintf_chk.c += $(libio-mtsafe)
CFLAGS-vsnprintf_chk.c += $(libio-mtsafe)
CFLAGS-asprintf_chk.c += $(libio-mtsafe)
CFLAGS-vasprintf_chk.c += $(libio-mtsafe)
CFLAGS-obprintf_chk.c += $(libio-mtsafe)
CFLAGS-dprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-vdprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-printf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-vprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-vfprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-gets_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fgets_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fgets_u_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fread_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fread_u_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-swprintf_chk.c += $(libio-mtsafe)
CFLAGS-vswprintf_chk.c += $(libio-mtsafe)
CFLAGS-wprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fwprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-vwprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-vfwprintf_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fgetws_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-fgetws_u_chk.c += $(libio-mtsafe) -fexceptions
CFLAGS-read_chk.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pread_chk.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pread64_chk.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-recv_chk.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-recvfrom_chk.c += -fexceptions -fasynchronous-unwind-tables

# Need to make sure the settings here override what configure might have
# set up for us, so keep the CFLAGS/CPPFLAGS split logical as the order is:
# <user CFLAGS> <test CFLAGS> <user CPPFLAGS> <test CPPFLAGS>
CFLAGS-tst-longjmp_chk.c += -fexceptions -fasynchronous-unwind-tables
CPPFLAGS-tst-longjmp_chk.c += -D_FORTIFY_SOURCE=1
CFLAGS-tst-longjmp_chk2.c += -fexceptions -fasynchronous-unwind-tables
CPPFLAGS-tst-longjmp_chk2.c += -D_FORTIFY_SOURCE=1
CFLAGS-tst-longjmp_chk3.c += -fexceptions -fasynchronous-unwind-tables
CPPFLAGS-tst-longjmp_chk3.c += -D_FORTIFY_SOURCE=1

# We know these tests have problems with format strings, this is what
# we are testing.  Disable that warning.  They are also testing
# deprecated functions (notably gets) so disable that warning as well.
# And they also generate warnings from warning attributes, which
# cannot be disabled via pragmas, so require -Wno-error to be used.
CFLAGS-tst-chk1.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-chk2.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-chk3.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-chk4.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-chk5.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-chk6.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk1.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk2.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk3.c += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk4.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk5.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
CFLAGS-tst-lfschk6.cc += -Wno-format -Wno-deprecated-declarations -Wno-error
LDLIBS-tst-chk4 = -lstdc++
LDLIBS-tst-chk5 = -lstdc++
LDLIBS-tst-chk6 = -lstdc++
LDLIBS-tst-lfschk4 = -lstdc++
LDLIBS-tst-lfschk5 = -lstdc++
LDLIBS-tst-lfschk6 = -lstdc++

# backtrace_symbols only works if we link with -rdynamic.  backtrace
# requires unwind tables on most architectures.
CFLAGS-tst-backtrace2.c += -funwind-tables
CFLAGS-tst-backtrace3.c += -funwind-tables
CFLAGS-tst-backtrace4.c += -funwind-tables
CFLAGS-tst-backtrace5.c += -funwind-tables
CFLAGS-tst-backtrace6.c += -funwind-tables
LDFLAGS-tst-backtrace2 = -rdynamic
LDFLAGS-tst-backtrace3 = -rdynamic
LDFLAGS-tst-backtrace4 = -rdynamic
LDFLAGS-tst-backtrace5 = -rdynamic
LDFLAGS-tst-backtrace6 = -rdynamic

CFLAGS-tst-ssp-1.c += -fstack-protector-all

tests = backtrace-tst tst-longjmp_chk tst-chk1 tst-chk2 tst-chk3 \
	tst-lfschk1 tst-lfschk2 tst-lfschk3 test-strcpy_chk test-stpcpy_chk \
	tst-chk4 tst-chk5 tst-chk6 tst-lfschk4 tst-lfschk5 tst-lfschk6 \
	tst-longjmp_chk2 tst-backtrace2 tst-backtrace3 tst-backtrace4 \
	tst-backtrace5 tst-backtrace6

ifeq ($(have-ssp),yes)
tests += tst-ssp-1
endif

ifeq (,$(CXX))
tests-unsupported = tst-chk4 tst-chk5 tst-chk6 \
		    tst-lfschk4 tst-lfschk5 tst-lfschk6
endif

extra-libs = libSegFault libpcprofile
extra-libs-others = $(extra-libs)

libSegFault-routines = segfault
libSegFault-inhibit-o = $(filter-out .os,$(object-suffixes))

libpcprofile-routines = pcprofile
libpcprofile-inhibit-o = $(filter-out .os,$(object-suffixes))

others = pcprofiledump
install-bin = pcprofiledump
install-bin-script = xtrace

ifeq ($(build-shared),yes)
install-bin-script += catchsegv
endif
generated += catchsegv xtrace

include ../Rules

ifeq ($(run-built-tests),yes)
LOCALES := de_DE.UTF-8
include ../gen-locales.mk

$(objpfx)tst-chk1.out: $(gen-locales)
$(objpfx)tst-chk2.out: $(gen-locales)
$(objpfx)tst-chk3.out: $(gen-locales)
$(objpfx)tst-chk4.out: $(gen-locales)
$(objpfx)tst-chk5.out: $(gen-locales)
$(objpfx)tst-chk6.out: $(gen-locales)
$(objpfx)tst-lfschk1.out: $(gen-locales)
$(objpfx)tst-lfschk2.out: $(gen-locales)
$(objpfx)tst-lfschk3.out: $(gen-locales)
$(objpfx)tst-lfschk4.out: $(gen-locales)
$(objpfx)tst-lfschk5.out: $(gen-locales)
$(objpfx)tst-lfschk6.out: $(gen-locales)
endif

sLIBdir := $(shell echo $(slibdir) | sed 's,lib\(\|64\)$$,\\\\$$LIB,')

$(objpfx)catchsegv: catchsegv.sh $(common-objpfx)soversions.mk \
		    $(common-objpfx)config.make
	slibpfx=`echo $(slibdir)|sed 's/lib\(64\|\)$$/\\\\\\\\$$LIB/'`; \
	sed -e 's|@VERSION@|$(version)|' -e "s|@SLIB@|$$slibpfx|" \
	    -e 's|@PKGVERSION@|$(PKGVERSION)|' \
	    -e 's|@REPORT_BUGS_TO@|$(REPORT_BUGS_TO)|' $< > $@.new
	chmod 555 $@.new
	mv -f $@.new $@

$(objpfx)pcprofiledump: $(objpfx)pcprofiledump.o

$(objpfx)xtrace: xtrace.sh
	rm -f $@.new
	sed -e 's|@VERSION@|$(version)|' -e 's|@SLIBDIR@|$(sLIBdir)|' \
	    -e 's|@BINDIR@|$(bindir)|' -e 's|@PKGVERSION@|$(PKGVERSION)|' \
	    -e 's|@REPORT_BUGS_TO@|$(REPORT_BUGS_TO)|' $^ > $@.new \
	&& rm -f $@ && mv $@.new $@ && chmod +x $@
