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

subdir		:= dlfcn

include ../Makeconfig

headers		:= bits/dlfcn.h dlfcn.h
extra-libs	:= libdl
libdl-routines	:= libdl-compat
routines = \
  dladdr \
  dladdr1 \
  dlclose \
  dlerror \
  dlinfo \
  dlmopen \
  dlopen \
  dlsym \
  dlvsym \
  libc_dlerror_result \

extra-libs-others := libdl

libdl-shared-only-routines += libdl-compat

# Pretend that libdl.so is a linker script, so that the symbolic
# link is not installed.
install-lib-ldscripts = libdl.so
$(inst_libdir)/libdl.so:

ifeq ($(build-shared),yes)
routines += dlopenold
shared-only-routines := dlopenold
endif

ifeq (yes,$(build-shared))
tests = glrefmain failtest tst-dladdr default errmsg1 tstcxaatexit \
	bug-dlopen1 bug-dlsym1 tst-dlinfo bug-atexit1 bug-atexit2 \
	bug-atexit3 tstatexit bug-dl-leaf tst-rec-dlopen
endif
modules-names = glreflib1 glreflib2 glreflib3 failtestmod defaultmod1 \
		defaultmod2 errmsg1mod modatexit modcxaatexit \
		bug-dlsym1-lib1 bug-dlsym1-lib2 bug-atexit1-lib \
		bug-atexit2-lib bug-dl-leaf-lib \
		bug-dl-leaf-lib-cb moddummy1 moddummy2

failtestmod.so-no-z-defs = yes
glreflib2.so-no-z-defs = yes
errmsg1mod.so-no-z-defs = yes

ifeq (yes,$(build-shared))
tests += tststatic tststatic2 tststatic3 tststatic4 tststatic5
tests-static += tststatic tststatic2 tststatic3 tststatic4 tststatic5
modules-names += modstatic modstatic2 modstatic3 modstatic5
tststatic-ENV = LD_LIBRARY_PATH=$(ld-library-path):$(common-objpfx)elf
tststatic2-ENV = $(tststatic-ENV)
tststatic3-ENV = $(tststatic-ENV)
tststatic4-ENV = $(tststatic-ENV)
tststatic5-ENV = $(tststatic-ENV)

ifneq (,$(CXX))
modules-names += bug-atexit3-lib
else
tests-unsupported += bug-atexit3
endif
endif

extra-test-objs += $(modules-names:=.os)
generated += $(modules-names:=.so)

include ../Rules

test-modules = $(addprefix $(objpfx),$(addsuffix .so,$(modules-names)))

ifeq ($(build-shared),yes)
# Build all the modules even when not actually running test programs.
tests: $(test-modules)
endif

$(objpfx)glrefmain.out: $(objpfx)glrefmain \
			$(objpfx)glreflib1.so $(objpfx)glreflib2.so

$(objpfx)failtest.out: $(objpfx)failtestmod.so

$(objpfx)tst-dladdr.out: $(objpfx)glreflib1.so

$(objpfx)tst-dlinfo.out: $(objpfx)glreflib3.so
LDFLAGS-glreflib3.so = -Wl,-rpath,:

LDFLAGS-default = $(LDFLAGS-rdynamic)
$(objpfx)default: $(objpfx)defaultmod1.so $(objpfx)defaultmod2.so
LDFLAGS-defaultmod2.so = $(LDFLAGS-Bsymbolic)

$(objpfx)errmsg1.out: $(objpfx)errmsg1 $(objpfx)errmsg1mod.so

$(objpfx)tstatexit.out: $(objpfx)tstatexit $(objpfx)modatexit.so

$(objpfx)tstcxaatexit.out: $(objpfx)tstcxaatexit $(objpfx)modcxaatexit.so

$(objpfx)tststatic.out: $(objpfx)tststatic $(objpfx)modstatic.so

$(objpfx)tststatic2.out: $(objpfx)tststatic2 $(objpfx)modstatic.so \
			 $(objpfx)modstatic2.so

$(objpfx)tststatic3.out: $(objpfx)tststatic3 $(objpfx)modstatic3.so

$(objpfx)tststatic4.out: $(objpfx)tststatic4 $(objpfx)modstatic3.so

$(objpfx)tststatic5.out: $(objpfx)tststatic5 $(objpfx)modstatic5.so

$(objpfx)bug-dlsym1: $(objpfx)bug-dlsym1-lib2.so
$(objpfx)bug-dlsym1.out: $(objpfx)bug-dlsym1-lib1.so \
			 $(objpfx)bug-dlsym1-lib2.so
$(objpfx)bug-dlsym1-lib1.so: $(objpfx)bug-dlsym1-lib2.so

$(objpfx)bug-atexit1.out: $(objpfx)bug-atexit1-lib.so

$(objpfx)bug-atexit2.out: $(objpfx)bug-atexit2-lib.so

ifneq (,$(CXX))
LDLIBS-bug-atexit3-lib.so = -lstdc++ -lgcc_eh
$(objpfx)bug-atexit3-lib.so: $(libsupport)
$(objpfx)bug-atexit3.out: $(objpfx)bug-atexit3-lib.so
endif

$(objpfx)bug-dl-leaf: $(objpfx)bug-dl-leaf-lib.so
$(objpfx)bug-dl-leaf.out: $(objpfx)bug-dl-leaf-lib-cb.so
$(objpfx)bug-dl-leaf-lib-cb.so: $(objpfx)bug-dl-leaf-lib.so

$(objpfx)tst-rec-dlopen.out: $(objpfx)moddummy1.so $(objpfx)moddummy2.so
