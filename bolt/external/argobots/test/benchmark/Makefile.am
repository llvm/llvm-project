# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

TESTS = \
	init_finalize \
	xstream_ops \
	thread_ops \
	thread_ops_all \
	thread_many_ops \
	thread_fork_join \
	thread_fork_join_many \
	thread_fork_join_many_priv_pool \
	task_fork_join \
	task_fork_join_priv_pool \
	task_ops \
	task_ops_all \
	sync_ops

if ABT_USE_PAPI
TESTS += \
	thread_fork_join_papi \
	thread_fork_join_papi_l1m_l2m \
	thread_fork_join_many_papi \
	thread_fork_join_many_papi_l1m_l2m \
	thread_fork_join_many_priv_pool_papi \
	thread_fork_join_many_priv_pool_papi_l1m_l2m \
	task_fork_join_papi \
	task_fork_join_papi_l1m_l2m \
	task_fork_join_priv_pool_papi \
	task_fork_join_priv_pool_papi_l1m_l2m
endif

check_PROGRAMS = $(TESTS)
noinst_PROGRAMS = $(TESTS)

include $(top_srcdir)/test/Makefile.mk

init_finalize_SOURCES = init_finalize.c
xstream_ops_SOURCES = xstream_ops.c
thread_ops_SOURCES = thread_ops.c
thread_ops_all_SOURCES = thread_ops_all.c
thread_many_ops_SOURCES = thread_many_ops.c
thread_fork_join_SOURCES =  thread_fork_join.c
thread_fork_join_many_SOURCES =  thread_fork_join.c
thread_fork_join_many_priv_pool_SOURCES =  thread_fork_join.c
task_fork_join_SOURCES =  task_fork_join.c
task_fork_join_priv_pool_SOURCES =  task_fork_join.c
task_ops_SOURCES = task_ops.c
task_ops_all_SOURCES = task_ops_all.c
sync_ops_SOURCES = sync_ops.c

thread_fork_join_many_CFLAGS = -DUSE_JOIN_MANY
thread_fork_join_many_priv_pool_CFLAGS = -DUSE_JOIN_MANY -DUSE_PRIV_POOL
task_fork_join_priv_pool_CFLAGS = -DUSE_PRIV_POOL

if ABT_USE_PAPI
thread_fork_join_papi_SOURCES = thread_fork_join.c
thread_fork_join_papi_l1m_l2m_SOURCES = thread_fork_join.c
thread_fork_join_many_papi_SOURCES = thread_fork_join.c
thread_fork_join_many_papi_l1m_l2m_SOURCES = thread_fork_join.c
thread_fork_join_many_priv_pool_papi_SOURCES = thread_fork_join.c
thread_fork_join_many_priv_pool_papi_l1m_l2m_SOURCES = thread_fork_join.c
task_fork_join_papi_SOURCES = task_fork_join.c
task_fork_join_papi_l1m_l2m_SOURCES = task_fork_join.c
task_fork_join_priv_pool_papi_SOURCES = task_fork_join.c
task_fork_join_priv_pool_papi_l1m_l2m_SOURCES = task_fork_join.c

thread_fork_join_papi_CFLAGS = -DUSE_PAPI @PAPI_CFLAGS@
thread_fork_join_papi_LDFLAGS = @PAPI_LDFLAGS@
thread_fork_join_papi_l1m_l2m_CFLAGS = -DUSE_PAPI -DUSE_PAPI_L1M_L2M @PAPI_CFLAGS@
thread_fork_join_papi_l1m_l2m_LDFLAGS = @PAPI_LDFLAGS@
thread_fork_join_many_papi_CFLAGS = -DUSE_JOIN_MANY -DUSE_PAPI @PAPI_CFLAGS@
thread_fork_join_many_papi_LDFLAGS = @PAPI_LDFLAGS@
thread_fork_join_many_papi_l1m_l2m_CFLAGS = -DUSE_JOIN_MANY -DUSE_PAPI -DUSE_PAPI_L1M_L2M @PAPI_CFLAGS@
thread_fork_join_many_papi_l1m_l2m_LDFLAGS = @PAPI_LDFLAGS@
thread_fork_join_many_priv_pool_papi_CFLAGS = -DUSE_JOIN_MANY -DUSE_PRIV_POOL -DUSE_PAPI @PAPI_CFLAGS@
thread_fork_join_many_priv_pool_papi_LDFLAGS = @PAPI_LDFLAGS@
thread_fork_join_many_priv_pool_papi_l1m_l2m_CFLAGS = -DUSE_JOIN_MANY -DUSE_PRIV_POOL -DUSE_PAPI -DUSE_PAPI_L1M_L2M @PAPI_CFLAGS@
thread_fork_join_many_priv_pool_papi_l1m_l2m_LDFLAGS = @PAPI_LDFLAGS@

task_fork_join_papi_CFLAGS = -DUSE_PAPI @PAPI_CFLAGS@
task_fork_join_papi_LDFLAGS = @PAPI_LDFLAGS@
task_fork_join_papi_l1m_l2m_CFLAGS = -DUSE_PAPI -DUSE_PAPI_L1M_L2M @PAPI_CFLAGS@
task_fork_join_papi_l1m_l2m_LDFLAGS = @PAPI_LDFLAGS@
task_fork_join_priv_pool_papi_CFLAGS = -DUSE_PRIV_POOL -DUSE_PAPI @PAPI_CFLAGS@
task_fork_join_priv_pool_papi_LDFLAGS = @PAPI_LDFLAGS@
task_fork_join_priv_pool_papi_l1m_l2m_CFLAGS = -DUSE_PRIV_POOL -DUSE_PAPI -DUSE_PAPI_L1M_L2M @PAPI_CFLAGS@
task_fork_join_priv_pool_papi_l1m_l2m_LDFLAGS = @PAPI_LDFLAGS@
endif

testing:
	./init_finalize -i 100
	./xstream_ops -e 4 -i 100
	./thread_ops -e 4 -u 10 -i 100
	./thread_ops_all -e 4 -u 10 -i 100
	./thread_many_ops -e 4 -u 10 -i 100
	./thread_fork_join -e 1 -u1024 -i 100
	./thread_fork_join_many -e 1 -u1024 -i 100
	./thread_fork_join_many_priv_pool -e 1 -u1024 -i 100
	./task_fork_join -e 1 -u1024 -i 100
	./task_fork_join_priv_pool -e 1 -u1024 -i 100
	./task_ops -e 4 -t 10 -i 100
	./task_ops_all -e 4 -t 10 -i 100
	./sync_ops -e 4 -u 10 -i 100
if ABT_USE_PAPI
	./thread_fork_join_papi -e 1 -u1024 -i 100
	./thread_fork_join_papi_l1m_l2m -e 1 -u1024 -i 100
	./thread_fork_join_many_papi -e 1 -u1024 -i 100
	./thread_fork_join_many_papi_l1m_l2m -e 1 -u1024 -i 100
	./thread_fork_join_many_priv_pool_papi -e 1 -u1024 -i 100
	./thread_fork_join_many_priv_pool_papi_l1m_l2m -e 1 -u1024 -i 100
	./task_fork_join_papi -e 1 -u1024 -i 100
	./task_fork_join_papi_l1m_l2m -e 1 -u1024 -i 100
	./task_fork_join_priv_pool_papi -e 1 -u1024 -i 100
	./task_fork_join_priv_pool_papi_l1m_l2m -e 1 -u1024 -i 100
endif
