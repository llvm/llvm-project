# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

TESTS = \
	stencil_barrier \
	stencil_depend_future \
	stencil_depend_mutex_cond \
	stencil_depend_yield \
	stencil_forkjoin \
	stencil_forkjoin_divconq \
	stencil_forkjoin_divconq_hrws \
	stencil_forkjoin_divconq_rws \
	stencil_forkjoin_revive \
	stencil_forkjoin_task \
	stencil_forkjoin_task_revive \
	stencil_forkjoin_ws \
	stencil_naive \
	stencil_seq

check_PROGRAMS = $(TESTS)
noinst_PROGRAMS = $(TESTS)

include $(top_srcdir)/examples/Makefile.mk

stencil_barrier_SOURCES = stencil_barrier.c
stencil_depend_future_SOURCES = stencil_depend_future.c
stencil_depend_mutex_cond_SOURCES = stencil_depend_mutex_cond.c
stencil_depend_yield_SOURCES = stencil_depend_yield.c
stencil_forkjoin_SOURCES = stencil_forkjoin.c
stencil_forkjoin_divconq_SOURCES = stencil_forkjoin_divconq.c
stencil_forkjoin_divconq_hrws_SOURCES = stencil_forkjoin_divconq_hrws.c
stencil_forkjoin_divconq_rws_SOURCES = stencil_forkjoin_divconq_rws.c
stencil_forkjoin_revive_SOURCES = stencil_forkjoin_revive.c
stencil_forkjoin_task_SOURCES = stencil_forkjoin_task.c
stencil_forkjoin_task_revive_SOURCES = stencil_forkjoin_task_revive.c
stencil_forkjoin_ws_SOURCES = stencil_forkjoin_ws.c
stencil_naive_SOURCES = stencil_naive.c
stencil_seq_SOURCES = stencil_seq.c
