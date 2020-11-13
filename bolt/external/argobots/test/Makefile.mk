# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS = $(DEPS_CPPFLAGS)
AM_CPPFLAGS += -I$(top_builddir)/src/include -I$(top_srcdir)/test/util
AM_LDFLAGS = $(DEPS_LDFLAGS) -lm

libabt = $(top_builddir)/src/libabt.la
libutil = $(top_builddir)/test/util/libutil.la

$(libabt):
	$(MAKE) -C $(top_builddir)/src

$(libutil):
	$(MAKE) -C $(top_builddir)/test/util

LDADD = $(libutil) $(libabt)

