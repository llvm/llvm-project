# -*- Mode: Makefile; -*-
#
# See LICENSE.txt in top-level directory.
#

CC=gcc
CFLAGS=-g -Wall -O2 `pkg-config --cflags argobots`
LDFLAGS=-lm `pkg-config --libs argobots`

PROGS = \
	parallel_for_abt_task \
	parallel_for_abt_thread \
	nested_parallel_for_abt_task \
	nested_parallel_for_abt_thread \
	nested_parallel_for_block_abt_thread \
	nested_parallel_for_block_abt_task \
	nested_parallel_for_irregular_abt_task \
	nested_parallel_for_irregular_abt_thread \
	task_single_producer_abt_task \
	task_single_producer_abt_thread \
	task_multiple_producer_abt_task \
	task_multiple_producer_abt_thread \
	task_nested_abt_task \
	task_nested_lvl2_abt_task

all: $(PROGS)

clean:
	rm -rf *.o *.dSYM $(PROGS)

testing:
	./parallel_for_abt_task
	./parallel_for_abt_thread
	./nested_parallel_for_abt_task
	./nested_parallel_for_abt_thread
	./nested_parallel_for_block_abt_thread
	./nested_parallel_for_block_abt_task
	./nested_parallel_for_irregular_abt_task
	./nested_parallel_for_irregular_abt_thread
	./task_single_producer_abt_task
	./task_single_producer_abt_thread
	./task_multiple_producer_abt_task
	./task_multiple_producer_abt_thread
	./task_nested_abt_task
	./task_nested_lvl2_abt_task

