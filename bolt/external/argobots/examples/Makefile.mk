# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS = $(DEPS_CPPFLAGS)
AM_CPPFLAGS += -I$(top_builddir)/src/include
AM_LDFLAGS = $(DEPS_LDFLAGS)

libabt = $(top_builddir)/src/libabt.la

$(libabt):
	$(MAKE) -C $(top_builddir)/src

LDADD = $(libabt)
