/*
 *  @copyright
 *  Copyright (C) 2009-2013, Intel Corporation
 *  All rights reserved.
 *  
 *  @copyright
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *    * Neither the name of Intel Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *  
 *  @copyright
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************
 *
 * cilk_undocumented.h
 *
 * This file defines exported functions that are not included in the standard
 * documentation.
 */

#ifndef INCLUDED_CILK_UNDOCUMENTED_H
#define INCLUDED_CILK_UNDOCUMENTED_H

#include <cilk/common.h>

#ifndef CILK_STUB

__CILKRTS_BEGIN_EXTERN_C

/*
 * __cilkrts_synched
 *
 * Allows an application to determine if there are any outstanding children at
 * this instant. This function will examine the current full frame to
 * determine this. This function will return a valid result only when called
 * within a spawn continuation, within the stack frame of the continuation
 * itself.
 */

CILK_EXPORT __CILKRTS_NOTHROW
int __cilkrts_synched(void);

/*
 * __cilkrts_cilkscreen_puts
 *
 * Allows an application to write a string to the Cilkscreen log.
 * The standard error stream will be flushed after the write.
 */

CILK_EXPORT __CILKRTS_NOTHROW
void __cilkrts_cilkscreen_puts(const char *);

/*
 * __cilkrts_get_sf
 *
 * A debugging aid that allows an application to get the __cilkrts_stack_frame
 * for the current function.  Only compiled into the DLL in debug builds.
 */

CILK_EXPORT __CILKRTS_NOTHROW
void *__cilkrts_get_sf(void);

/**
 * Returns the size of stacks created by Cilk.
 */
CILK_EXPORT __CILKRTS_NOTHROW
size_t __cilkrts_get_stack_size(void);

/** 
 * Dumps runtime statistics to stderr.
 * Undocumented API for debugging. 
 */
CILK_EXPORT __CILKRTS_NOTHROW
void __cilkrts_dump_stats(void);

CILK_EXPORT __CILKRTS_NOTHROW
int __cilkrts_irml_version(void);

struct __cilk_tbb_unwatch_thunk;
struct __cilk_tbb_stack_op_thunk;

CILK_EXPORT __CILKRTS_NOTHROW
int __cilkrts_watch_stack(struct __cilk_tbb_unwatch_thunk *u,
                          struct __cilk_tbb_stack_op_thunk o);

#ifndef IN_CILK_RUNTIME
#ifdef _WIN32
/* Do not use CILK_API because __cilkrts_worker_stub must be __stdcall */
CILK_EXPORT unsigned __CILKRTS_NOTHROW __stdcall
__cilkrts_worker_stub(void *arg);
#else
/* Do not use CILK_API because __cilkrts_worker_stub have default visibility */
CILK_EXPORT void* __CILKRTS_NOTHROW
__cilkrts_worker_stub(void *arg);
#endif /* _WIN32 */
#endif /* IN_CILK_RUNTIME */

__CILKRTS_END_EXTERN_C

#else /* CILK_STUB */

/* Stubs for the api functions */

#define __cilkrts_get_stack_size() (0)
#define __cilkrts_synched() (1)

#endif /* CILK_STUB */

#endif /* INCLUDED_CILK_UNDOCUMENTED_H */
