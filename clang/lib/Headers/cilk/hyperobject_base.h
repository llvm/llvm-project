/*
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
 *
 */

#ifndef INCLUDED_CILK_HYPEROBJECT_BASE
#define INCLUDED_CILK_HYPEROBJECT_BASE

#ifdef __cplusplus
# include <cstdlib>
# include <cstddef>
#else
# include <stdlib.h>
# include <stddef.h>
#endif

#include <cilk/common.h>

#if defined _WIN32 || defined _WIN64
# if !defined CILK_STUB && !defined IN_CILK_RUNTIME
    /* bring in the Intel(R) Cilk(TM) Plus library, which has definitions for some of these
     * functions. */
#   pragma comment(lib, "cilkrts")
# endif
#endif

/* The __CILKRTS_STRAND_PURE attribute tells the compiler that the value
 * returned by 'func' for a given argument to 'func' will remain valid until
 * the next strand boundary (spawn or sync) or until the next call to a
 * function with the __CILKRTS_STRAND_STALE attribute using the same function
 * argument.
 */
#if 0 && defined __cilk && (defined __GNUC__ && !defined _WIN32) && defined __cilkartsrev
# define __CILKRTS_STRAND_PURE(func) \
    func __attribute__((__cilk_hyper__("lookup")))
# define __CILKRTS_STRAND_STALE(func) \
    func __attribute__((__cilk_hyper__("flush")))
#else
# define __CILKRTS_STRAND_PURE(func) func
# define __CILKRTS_STRAND_STALE(func) func
#endif

/*****************************************************************************
 * C runtime interface to the hyperobject subsystem
 *****************************************************************************/

__CILKRTS_BEGIN_EXTERN_C

/* Callback function signatures.  The 'r' argument always points to the
 * reducer itself and is commonly ignored. */
typedef void (*cilk_c_reducer_reduce_fn_t)(void* r, void* lhs, void* rhs);
typedef void (*cilk_c_reducer_identity_fn_t)(void* r, void* view);
typedef void (*cilk_c_reducer_destroy_fn_t)(void* r, void* view);
typedef void* (*cilk_c_reducer_allocate_fn_t)(void* r, __STDNS size_t bytes);
typedef void (*cilk_c_reducer_deallocate_fn_t)(void* r, void* view);

/** Representation of the monoid */
typedef struct cilk_c_monoid {
    cilk_c_reducer_reduce_fn_t          reduce_fn;
    cilk_c_reducer_identity_fn_t        identity_fn;
    cilk_c_reducer_destroy_fn_t         destroy_fn;
    cilk_c_reducer_allocate_fn_t        allocate_fn;
    cilk_c_reducer_deallocate_fn_t      deallocate_fn;
} cilk_c_monoid;

/** Base of the hyperobject */
typedef struct __cilkrts_hyperobject_base
{
    cilk_c_monoid       __c_monoid;
    unsigned long long  __flags;
    __STDNS ptrdiff_t   __view_offset;  /* offset (in bytes) to leftmost view */
    __STDNS size_t      __view_size;    /* Size of each view */
} __cilkrts_hyperobject_base;


#ifndef CILK_STUB

/* Library functions. */
CILK_EXPORT
    void __cilkrts_hyper_create(__cilkrts_hyperobject_base *key);
CILK_EXPORT void __CILKRTS_STRAND_STALE(
    __cilkrts_hyper_destroy(__cilkrts_hyperobject_base *key));
CILK_EXPORT void* __CILKRTS_STRAND_PURE(
    __cilkrts_hyper_lookup(__cilkrts_hyperobject_base *key));

CILK_EXPORT
    void* __cilkrts_hyperobject_alloc(void* ignore, __STDNS size_t bytes);
CILK_EXPORT
    void __cilkrts_hyperobject_dealloc(void* ignore, void* view);

/* No-op destroy function */
CILK_EXPORT
    void __cilkrts_hyperobject_noop_destroy(void* ignore, void* ignore2);


#else // CILK_STUB

// Programs compiled with CILK_STUB are not linked with the Intel Cilk Plus runtime 
// library, so they should not have external references to cilkrts functions.
// Furthermore, they don't need the hyperobject functionality, so the
// functions can be stubbed.

#define __cilkrts_hyperobject_create __cilkrts_hyperobject_create__stub
__CILKRTS_INLINE
    void __cilkrts_hyper_create(__cilkrts_hyperobject_base *key) 
    {}

#define __cilkrts_hyperobject_destroy __cilkrts_hyperobject_destroy__stub
__CILKRTS_INLINE
    void __cilkrts_hyper_destroy(__cilkrts_hyperobject_base *key) 
    {}

#define __cilkrts_hyperobject_lookup __cilkrts_hyperobject_lookup__stub
__CILKRTS_INLINE
    void* __cilkrts_hyper_lookup(__cilkrts_hyperobject_base *key)
    { return (char*)(key) + key->__view_offset; }

// Pointers to these functions are stored into monoids, so real functions
// are needed.

#define __cilkrts_hyperobject_alloc __cilkrts_hyperobject_alloc__stub
__CILKRTS_INLINE
    void* __cilkrts_hyperobject_alloc(void* ignore, __STDNS size_t bytes)
    { assert(0); return __STDNS malloc(bytes); }

#define __cilkrts_hyperobject_dealloc __cilkrts_hyperobject_dealloc__stub
__CILKRTS_INLINE
    void __cilkrts_hyperobject_dealloc(void* ignore, void* view)
    { assert(0); __STDNS free(view); }

#define __cilkrts_hyperobject_noop_destroy \
            __cilkrts_hyperobject_noop_destroy__stub
__CILKRTS_INLINE
    void __cilkrts_hyperobject_noop_destroy(void* ignore, void* ignore2)
    {}
    
#endif

__CILKRTS_END_EXTERN_C

#endif /* INCLUDED_CILK_HYPEROBJECT_BASE */
