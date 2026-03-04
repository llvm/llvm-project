//===--------- ripple/thread.h - Declare ripple thread helpers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef RIPPLE_PARALLEL_STRINGIFY
#define RIPPLE_PARALLEL_STRINGIFY(x) #x
#endif

/**
 * A descriptive loop annotation to split a loop across the specified
 * thread dimension(s) work-items.
 */
#define ripple_parallel_thd(BlockShape, ...)                                   \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape) Dims(    \
      __VA_ARGS__) IgnoreNullStmts Schedule(static)))

/**
 * Same as 'ripple_parallel_thd' but dynamically dispatched using an iteration
 * server.
 */
#define ripple_parallel_thd_dyn(BlockShape, ...)                               \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape) Dims(    \
      __VA_ARGS__) IgnoreNullStmts Schedule(dynamic)))

/**
 * A descriptive loop annotation to split a loop across the specified
 * thread dimension(s) work-items. The minimum amount of work to distribute to
 * each thread is specified using Chunk (integer literal or variable).
 */
#define ripple_parallel_thd_chunk(BlockShape, Chunk, ...)                      \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape) Dims(    \
      __VA_ARGS__) IgnoreNullStmts ThreadChunk(Chunk)))

/**
 * Same as 'ripple_parallel_thd_chunk' but dynamically dispatched using an
 * iteration server.
 */
#define ripple_parallel_thd_chunk_dyn(BlockShape, Chunk, ...)                  \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape) Dims(    \
      __VA_ARGS__) IgnoreNullStmts ThreadChunk(Chunk) Schedule(dynamic)))

// Delegate to the ripple/thread.h from the runtime
#if __has_include_next(<ripple/thread.h>)

#include_next <ripple/thread.h>

#else

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ripple_thread_block *ripple_thd_block_t;

extern ripple_thd_block_t ripple_thd_init(int, void *);
extern void ripple_thd_exit(ripple_thd_block_t);
extern ripple_thd_block_t ripple_thd_set_block_shape(void *, unsigned, ...);
extern __SIZE_TYPE__ ripple_thd_id(ripple_thd_block_t, int);
extern __SIZE_TYPE__ ripple_thd_get_block_size(ripple_thd_block_t, int);
extern void ripple_thd_barrier(ripple_thd_block_t, unsigned);
extern int ripple_thd_is_main(ripple_thd_block_t, unsigned);

typedef struct {
  long value;
  long has_value;
} ripple_opt_it;
extern void ripple_it_serv_init(ripple_thd_block_t, unsigned int,
                                __INT32_TYPE__, __INT32_TYPE__, __INT32_TYPE__);
extern void ripple_it_serv_exit(ripple_thd_block_t, unsigned int, int);
extern ripple_opt_it ripple_it_serv_next(ripple_thd_block_t, unsigned int);

#ifdef __cplusplus
} // extern "C"
#endif

#endif