/* cilkscreen.h                  -*-C++-*-
 *
 *************************************************************************
 *
 * Copyright (C) 2010-2016, Intel Corporation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * *********************************************************************
 * 
 * PLEASE NOTE: This file is a downstream copy of a file mainitained in
 * a repository at cilkplus.org. Changes made to this file that are not
 * submitted through the contribution process detailed at
 * http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 * time that a new version is released. Changes only submitted to the
 * GNU compiler collection or posted to the git repository at
 * https://bitbucket.org/intelcilkruntime/itnel-cilk-runtime.git are
 * not tracked.
 * 
 * We welcome your contributions to this open source project. Thank you
 * for your assistance in helping us improve Cilk Plus.
 *
 **************************************************************************/

#ifndef INCLUDED_CILKSCREEN_H
#define INCLUDED_CILKSCREEN_H

#include <cilk/cilk_api.h>

/*
 * Cilkscreen "functions".  These macros generate metadata in your application
 * to notify Cilkscreen of program state changes
 */

#if ! defined(CILK_STUB) && defined(__INTEL_COMPILER)
#  define __cilkscreen_metacall(annotation,expr) \
    __notify_zc_intrinsic((char *)annotation, expr)
#else
#  define __cilkscreen_metacall(annotation,expr) ((void)annotation, (void)(expr))
#endif

/* Call once when a user thread enters a spawning function */
#define __cilkscreen_enable_instrumentation() \
    __cilkscreen_metacall("cilkscreen_enable_instrumentation", 0)

/* Call once when a user thread exits a spawning function */
#define  __cilkscreen_disable_instrumentation() \
    __cilkscreen_metacall("cilkscreen_disable_instrumentation", 0)

/* Call to temporarily disable cilkscreen instrumentation */
#define __cilkscreen_enable_checking() \
    __cilkscreen_metacall("cilkscreen_enable_checking", 0)

/* Call to re-enable temporarily-disabled cilkscreen instrumentation */
#define __cilkscreen_disable_checking() \
    __cilkscreen_metacall("cilkscreen_disable_checking", 0)

/* Inform cilkscreen that memory from begin to end can be reused without
 * causing races (e.g., for memory that comes from a memory allocator) */
#define __cilkscreen_clean(begin, end)                      \
    do {                                                    \
        void *__data[2] = { (begin), (end) };               \
        __cilkscreen_metacall("cilkscreen_clean", &__data); \
    } while(0)

/* Inform cilkscreen that a lock is being acquired.
 * If the lock type is not a handle, then the caller should take its address
 * and pass the pointer to the lock.  Otherwise, the caller should pass the
 * lock handle directly.
 */
#define __cilkscreen_acquire_lock(lock) \
    __cilkscreen_metacall("cilkscreen_acquire_lock", (lock))

#define __cilkscreen_release_lock(lock) \
    __cilkscreen_metacall("cilkscreen_release_lock", (lock))

/*
 * Metacall data
 *
 * A metacall is a way to pass data to a function implemented by a tool.
 * Metacalls are always instrumented when the tool is loaded.
 */

// Tool code for Cilkscreen
#define METACALL_TOOL_CILKSCREEN 1

// Metacall codes implemented by Cilkscreen
#define CS_METACALL_PUTS 0  // Write string to the Cilkscreen log

#define __cilkscreen_puts(text) \
    __cilkrts_metacall(METACALL_TOOL_CILKSCREEN, CS_METACALL_PUTS, (void *)(const char *)text)

#endif /* defined(INCLUDED_CILKSCREEN_H) */
