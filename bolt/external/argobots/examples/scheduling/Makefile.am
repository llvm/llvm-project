# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

TESTS = \
	sched_and_pool_user \
	sched_predef \
	sched_shared_pool \
	sched_stack \
	sched_user

check_PROGRAMS = $(TESTS)
noinst_PROGRAMS = $(TESTS)

include $(top_srcdir)/examples/Makefile.mk

sched_and_pool_user_SOURCES = sched_and_pool_user.c
sched_predef_SOURCES = sched_predef.c
sched_shared_pool_SOURCES = sched_shared_pool.c
sched_stack_SOURCES = sched_stack.c
sched_user_SOURCES = sched_user.c
