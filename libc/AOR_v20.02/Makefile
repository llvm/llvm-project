# Makefile - requires GNU make
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir = .
prefix = /usr
bindir = $(prefix)/bin
libdir = $(prefix)/lib
includedir = $(prefix)/include

# Configure these in config.mk, do not make changes in this file.
SUBS = math string networking
HOST_CC = cc
HOST_CFLAGS = -std=c99 -O2
HOST_LDFLAGS =
HOST_LDLIBS =
EMULATOR =
CPPFLAGS =
CFLAGS = -std=c99 -O2
CFLAGS_SHARED = -fPIC
CFLAGS_ALL = -Ibuild/include $(CPPFLAGS) $(CFLAGS)
LDFLAGS =
LDLIBS =
AR = $(CROSS_COMPILE)ar
RANLIB = $(CROSS_COMPILE)ranlib
INSTALL = install

all:

-include config.mk

$(foreach sub,$(SUBS),$(eval include $(srcdir)/$(sub)/Dir.mk))

# Required targets of subproject foo:
#   all-foo
#   check-foo
#   clean-foo
#   install-foo
# Required make variables of subproject foo:
#   foo-files: Built files (all in build/).
# Make variables used by subproject foo:
#   foo-...: Variables defined in foo/Dir.mk or by config.mk.

all: $(SUBS:%=all-%)

ALL_FILES = $(foreach sub,$(SUBS),$($(sub)-files))
DIRS = $(sort $(patsubst %/,%,$(dir $(ALL_FILES))))
$(ALL_FILES): | $(DIRS)
$(DIRS):
	mkdir -p $@

$(filter %.os,$(ALL_FILES)): CFLAGS_ALL += $(CFLAGS_SHARED)

build/%.o: $(srcdir)/%.S
	$(CC) $(CFLAGS_ALL) -c -o $@ $<

build/%.o: $(srcdir)/%.c
	$(CC) $(CFLAGS_ALL) -c -o $@ $<

build/%.os: $(srcdir)/%.S
	$(CC) $(CFLAGS_ALL) -c -o $@ $<

build/%.os: $(srcdir)/%.c
	$(CC) $(CFLAGS_ALL) -c -o $@ $<

clean: $(SUBS:%=clean-%)
	rm -rf build

distclean: clean
	rm -f config.mk

$(DESTDIR)$(bindir)/%: build/bin/%
	$(INSTALL) -D $< $@

$(DESTDIR)$(libdir)/%.so: build/lib/%.so
	$(INSTALL) -D $< $@

$(DESTDIR)$(libdir)/%: build/lib/%
	$(INSTALL) -m 644 -D $< $@

$(DESTDIR)$(includedir)/%: build/include/%
	$(INSTALL) -m 644 -D $< $@

install: $(SUBS:%=install-%)

check: $(SUBS:%=check-%)

.PHONY: all clean distclean install check
