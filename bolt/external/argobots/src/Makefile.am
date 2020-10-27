# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

abt_sources = \
	barrier.c \
	cond.c \
	error.c \
	eventual.c \
	futures.c \
	global.c \
	info.c \
	key.c \
	local.c \
	log.c \
	mutex.c \
	mutex_attr.c \
	rwlock.c \
	self.c \
	stream.c \
	stream_barrier.c \
	task.c \
	thread.c \
	thread_attr.c \
	timer.c \
	tool.c \
	unit.c \
	ythread.c \
	ythread_htable.c

include $(top_srcdir)/src/arch/Makefile.mk
include $(top_srcdir)/src/mem/Makefile.mk
include $(top_srcdir)/src/pool/Makefile.mk
include $(top_srcdir)/src/sched/Makefile.mk
include $(top_srcdir)/src/util/Makefile.mk

include $(top_srcdir)/src/include/Makefile.mk

lib_LTLIBRARIES = libabt.la
libabt_la_SOURCES = $(abt_sources)
libabt_la_CPPFLAGS = -I$(top_srcdir)/src/include @ABT_VISIBILITY_CFLAGS@
libabt_la_CCASFLAGS = -I$(top_srcdir)/src/include @ABT_VISIBILITY_CFLAGS@
libabt_la_LDFLAGS = -version-info @libabt_so_version@

