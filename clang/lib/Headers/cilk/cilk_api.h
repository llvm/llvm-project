/*  cilk_api.h
 *
 *  Copyright (C) 2009-2018, Intel Corporation
 *  All rights reserved.
 *  
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
 *  *********************************************************************
 *  
 *  PLEASE NOTE: This file is a downstream copy of a file maintained in
 *  a repository at cilkplus.org. Changes made to this file that are not
 *  submitted through the contribution process detailed at
 *  http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 *  time that a new version is released. Changes only submitted to the
 *  GNU compiler collection or posted to the git repository at
 *  https://bitbucket.org/intelcilkruntime/intel-cilk-runtime are
 *  not tracked.
 *  
 *  We welcome your contributions to this open source project. Thank you
 *  for your assistance in helping us improve Cilk Plus.
 */

/** @file cilk_api.h
 *
 * @brief Defines the Intel(R) Cilk(TM) Plus API for use by applications.
 *
 *  @ingroup api
 */

#ifndef INCLUDED_CILK_API_H
#define INCLUDED_CILK_API_H


/** @defgroup api Runtime API
* API to interact with the Intel Cilk Plus runtime.
* @{
*/

#ifndef CILK_STUB /* Real (non-stub) definitions */

#if ! defined(__cilk) && ! defined(USE_CILK_API)
#   ifdef _WIN32
#       error Cilk API is being used with non-Cilk compiler (or Cilk is disabled)
#   else
#       warning Cilk API is being used with non-Cilk compiler (or Cilk is disabled)
#   endif
#endif

#include <cilk/common.h>

#ifdef __cplusplus
#   include <cstddef>  /* Defines size_t */
#else
#   include <stddef.h> /* Defines size_t */
#endif

#ifdef _WIN32
#   ifndef IN_CILK_RUNTIME
/* Ensure the library is brought if any of these functions are being called. */
#       pragma comment(lib, "cilkrts")
#   endif

#   ifndef __cplusplus
#       include <wchar.h>
#   endif
#endif /* _WIN32 */

__CILKRTS_BEGIN_EXTERN_C

/** Return values from `__cilkrts_set_param()` and `__cilkrts_set_param_w()`
 */
enum __cilkrts_set_param_status {
    __CILKRTS_SET_PARAM_SUCCESS = 0, /**< Success - parameter set */
    __CILKRTS_SET_PARAM_UNIMP   = 1, /**< Unimplemented parameter */
    __CILKRTS_SET_PARAM_XRANGE  = 2, /**< Parameter value out of range */
    __CILKRTS_SET_PARAM_INVALID = 3, /**< Invalid parameter value */
    __CILKRTS_SET_PARAM_LATE    = 4  /**< Too late to change parameter value */
};

/** Sets user controllable runtime parameters
 *
 *  Call this function to set runtime parameters that control the behavior
 *  of the Intel Cilk Plus scheduler.
 *
 *  @param param    A string specifying the parameter to be set. One of:
 *  -   `"nworkers"`
 *  -   `"force reduce"`
 *  @param value    A string specifying the parameter value.
 *  @returns        A value from the @ref __cilkrts_set_param_status
 *                  enumeration indicating the result of the operation.
 *
 *  @par The "nworkers" parameter
 *
 *  This parameter specifies the number of worker threads to be created by the
 *  Intel Cilk Plus runtime. @a Value must be a string of digits to be parsed by
 *  `strtol()` as a decimal number.
 *
 *  The number of worker threads is:
 *  1.  the value set with `__cilkrts_set_param("nworkers")`, if it is
 *      positive; otherwise,
 *  2.  the value of the CILK_NWORKERS environment variable, if it is
 *      defined; otherwise
 *  3.  the number of cores available, as reported by the operating system.
 *
 *  @note
 *  Technically, Intel Cilk Plus distinguishes between the _user thread_ (the thread
 *  that the user code was executing on when the Intel Cilk Plus runtime started),
 *  and _worker threads_ (new threads created by the Intel Cilk Plus runtime to
 *  support Intel Cilk Plus parallelism). `nworkers` actually includes both the user
 *  thread and the worker threads; that is, it is one greater than the number of
 *  true "worker threads".
 *
 *  @note
 *  Setting `nworkers = 1` produces serial behavior. Intel Cilk Plus spawns and syncs
 *  will be executed, but with only one worker, continuations will never be
 *  stolen, so all code will execute in serial.
 *
 *  @warning
 *  The number of worker threads can only be set *before* the runtime has
 *  started. Attempting to set it when the runtime is running will have no
 *  effect, and will return an error code. You can call __cilkrts_end_cilk()
 *  to shut down the runtime to change the number of workers.
 *
 *  @warning
 *  The default Intel Cilk scheduler behavior is usually pretty good. The
 *  ability to override `nworkers` can be useful for experimentation, but it
 *  won't usually be necessary for getting good performance.
 *
 *  @par The "force reduce" parameter
 *
 *  This parameter controls whether the runtime should allocate a new view
 *  for a reducer for every parallel strand that it is accessed on. (See
 *  @ref pagereducers.) @a Value must be `"1"` or `"true"` to enable the
 *  "force reduce" behavior, or `"0"` or `"false"` to disable it.
 *
 *  "Force reduce" behavior will also be enabled if
 *  `__cilkrts_set_param("force reduce")` is not called, but the
 *  `CILK_FORCE_REDUCE` environment variable is defined.
 *
 *  @warning
 *  When this option is enabled, `nworkers` should be set to `1`. Using "force
 *  reduce" with more than one worker may result in runtime errors.
 *
 *  @warning
 *  Enabling this option can significantly reduce performance. Use it
 *  _only_ as a debugging tool.
 */
CILK_API(int) __cilkrts_set_param(const char *param, const char *value);

#ifdef _WIN32
/**
 * Sets user controllable parameters using wide strings
 *
 * @note This variant of __cilkrts_set_param() is only available
 * on Windows.
 *
 * @copydetails __cilkrts_set_param
 */
CILK_API(int) __cilkrts_set_param_w(const wchar_t *param, const wchar_t *value);
#endif

/** Shuts down and deallocates all Intel Cilk Plus states. If Intel Cilk Plus is still in
 * use by the calling thread, the runtime aborts the application. Otherwise, the
 * runtime waits for all other threads using Intel Cilk Plus to exit.
 */
CILK_API(void) __cilkrts_end_cilk(void);

/** Initializes Intel Cilk Plus data structures and start the runtime.
 */
CILK_API(void) __cilkrts_init(void);

/** Returns the runtime `nworkers` parameter. (See the discussion of `nworkers`
 *  in the documentation for __cilkrts_set_param().)
 */
CILK_API(int) __cilkrts_get_nworkers(void);

/** Returns the number of thread data structures.
 *
 *  This function returns the number of data structures that have been allocated
 *  by the runtime to hold information about user and worker threads.
 *
 *  If you don't already know what this is good for, then you probably don't
 *  need it. :)
 */
CILK_API(int) __cilkrts_get_total_workers(void);

/** Returns a small integer identifying the current thread.
 *
 *  What thread is the function running on? Each worker thread
 *  started by the Intel Cilk Plus runtime library has a unique worker number in the
 *  range `1 .. nworkers - 1`.
 *
 *  All _user_ threads (threads started by the user, or by other libraries) are
 *  identified as worker number 0. Therefore, the worker number is not unique
 *  across multiple user threads.
 */
CILK_API(int) __cilkrts_get_worker_number(void);

/** Tests whether "force reduce" behavior is enabled.
 *
 *  @return Non-zero if force-reduce mode is on, zero if it is off.
 */
CILK_API(int) __cilkrts_get_force_reduce(void);

/** Interacts with tools
 */
CILK_API(void)
    __cilkrts_metacall(unsigned int tool, unsigned int code, void *data);

#ifdef _WIN32
/// Windows exception description record.
typedef struct _EXCEPTION_RECORD _EXCEPTION_RECORD;

/** Function signature for Windows exception notification callbacks.
 */
typedef void (*__cilkrts_pfn_seh_callback)(const _EXCEPTION_RECORD *exception);

/** Specifies a function to call when a non-C++ exception is caught.
 *
 *  Intel Cilk Plus parallelism plays nicely with C++ exception handling, but
 *  the Intel Cilk Plus runtime has no way to unwind the stack across a strand
 *  boundary for Microsoft SEH ("Structured Exception Handling") exceptions.
 *  Therefore, when the runtime catches such an exception, it must abort the
 *  application.
 *
 *  If an SEH callback has been set, the runtime will call it before aborting.
 *
 *  @param  pfn A pointer to a callback function to be called before the
 *              runtime aborts the program because of an SEH exception.
 */
CILK_API(int) __cilkrts_set_seh_callback(__cilkrts_pfn_seh_callback pfn);
#endif /* _WIN32 */

#if __CILKRTS_ABI_VERSION >= 1
/* Pedigree API is available only for compilers that use ABI version >= 1. */


/** @name Pedigrees
 */
//@{

// @cond internal

/** Support for __cilkrts_get_pedigree.
 */
CILK_API(__cilkrts_pedigree)
__cilkrts_get_pedigree_internal(__cilkrts_worker *w);

/** Support for __cilkrts_bump_worker_rank.
 */
CILK_API(int)
__cilkrts_bump_worker_rank_internal(__cilkrts_worker* w);

/// @endcond


/** Gets the current pedigree in a linked list representation.
 *
 *  This routine returns a copy of the last node in the pedigree list.
 *  For example, if the current pedigree (in order) is <1, 2, 3, 4>,
 *  then this method returns a node with rank == 4, and whose parent
 *  field points to the node with rank of 3.  In summary, following the
 *  nodes in the chain visits the terms of the pedigree in reverse.
 *
 *  The returned node is guaranteed to be valid only until the caller
 *  of this routine has returned.
 */
__CILKRTS_INLINE
__cilkrts_pedigree __cilkrts_get_pedigree(void)
{
    return __cilkrts_get_pedigree_internal(__cilkrts_get_tls_worker());
}

/** Context used by __cilkrts_get_pedigree_info.
 *
 *  @deprecated
 *  This data structure is only used by the deprecated
 *  __cilkrts_get_pedigree_info function.
 *
 *  Callers should initialize the `data` array to NULL and set the `size`
 *  field to `sizeof(__cilkrts_pedigree_context_t)` before the first call
 *  to `__cilkrts_get_pedigree_info()`. Also, callers should not examine or
 *  modify `data` thereafter.
 */
typedef struct
{
    __STDNS size_t size;    /**< Size of the struct in bytes */
    void *data[3];          /**< Opaque context data */
} __cilkrts_pedigree_context_t;

/** Gets pedigree information.
 *
 *  @deprecated
 *  Use __cilkrts_get_pedigree() instead.
 *
 *  This routine allows code to walk up the stack of Intel Cilk Plus frames to gather
 *  the pedigree.
 *
 *  Initialize the pedigree walk by filling the pedigree context with NULLs
 *  and setting the size field to `sizeof(__cilkrts_pedigree_context)`.
 *  Other than initialization to NULL to start the walk, user coder should
 *  consider the pedigree context data opaque and should not examine or
 *  modify it.
 *
 * @returns  0 - Success - birthrank is valid
 * @returns >0 - End of pedigree walk
 * @returns -1 - Failure - No worker bound to thread
 * @returns -2 - Failure - Sanity check failed,
 * @returns -3 - Failure - Invalid context size
 * @returns -4 - Failure - Internal error - walked off end of chain of frames
 */
CILK_API(int)
__cilkrts_get_pedigree_info(/* In/Out */ __cilkrts_pedigree_context_t *context,
                            /* Out */    uint64_t *sf_birthrank);

/** Gets the rank of the currently executing worker.
 *
 *  @deprecated
 *  Use `__cilkrts_get_pedigree().rank` instead.
 *
 * @returns  0 - Success - *rank is valid
 * @returns <0 - Failure - *rank is not changed
 */
CILK_EXPORT_AND_INLINE
int __cilkrts_get_worker_rank(uint64_t *rank)
{
    *rank = __cilkrts_get_pedigree().rank;
    return 0;
}

/** Increments the pedigree rank of the currently executing worker.
 *
 * @returns 0 - Success - rank was incremented
 * @returns -1 - Failure
 */
CILK_EXPORT_AND_INLINE
int __cilkrts_bump_worker_rank(void)
{
    return __cilkrts_bump_worker_rank_internal(__cilkrts_get_tls_worker());
}

/** Increments the pedigree rank for a `cilk_for` loop.
 *  Obsolete.
 *
 *  @deprecated
 *  This function was provided to allow the user to manipulate the pedigree
 *  rank of a `cilk_for` loop. The compiler now generates code to do that
 *  manipulation automatically, so this function is now unnecessary. It may
 *  be called, but will have no effect.
 */
CILK_EXPORT_AND_INLINE
int __cilkrts_bump_loop_rank(void)
{
    return 0;
}

//@}

#endif /* __CILKRTS_ABI_VERSION >= 1 */

__CILKRTS_END_EXTERN_C

#else /* CILK_STUB */

// Programs compiled with CILK_STUB are not linked with the Intel Cilk Plus runtime
// library, so they should not have external references to runtime functions.
// Therefore, the functions are replaced with stubs.

#ifdef _WIN32
#define __cilkrts_set_param_w(name,value) ((value), 0)
#define __cilkrts_set_seh_callback(pfn) (0)
#endif
#define __cilkrts_set_param(name,value) ((value), 0)
#define __cilkrts_end_cilk() ((void) 0)
#define __cilkrts_init() ((void) 0)
#define __cilkrts_get_nworkers() (1)
#define __cilkrts_get_total_workers() (1)
#define __cilkrts_get_worker_number() (0)
#define __cilkrts_get_force_reduce() (0)
#define __cilkrts_metacall(tool,code,data) ((tool), (code), (data), 0)

#if __CILKRTS_ABI_VERSION >= 1
/* Pedigree stubs */
#define __cilkrts_get_pedigree_info(context, sf_birthrank) (-1)
#define __cilkrts_get_worker_rank(rank) (*(rank) = 0)
#define __cilkrts_bump_worker_rank() (-1)
#define __cilkrts_bump_loop_rank() (-1)

/*
 * A stub method for __cilkrts_get_pedigree.
 * Returns an empty __cilkrts_pedigree.
 */
__CILKRTS_INLINE
__cilkrts_pedigree __cilkrts_get_pedigree_stub(void)
{
    __cilkrts_pedigree ans;
    ans.rank = 0;
    ans.parent = NULL;
    return ans;
}

/* Renamed to an actual stub method. */
#define __cilkrts_get_pedigree() __cilkrts_get_pedigree_stub()

#endif /* __CILKRTS_ABI_VERSION >= 1 */

#endif /* CILK_STUB */

//@}

#endif /* INCLUDED_CILK_API_H */
