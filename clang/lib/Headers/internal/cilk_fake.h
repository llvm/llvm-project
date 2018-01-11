/* cilk_fake.h                  -*-C++-*-
 *
 *************************************************************************
 *
 *  Copyright (C) 2011-2016, Intel Corporation
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
 *  PLEASE NOTE: This file is a downstream copy of a file mainitained in
 *  a repository at cilkplus.org. Changes made to this file that are not
 *  submitted through the contribution process detailed at
 *  http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 *  time that a new version is released. Changes only submitted to the
 *  GNU compiler collection or posted to the git repository at
 *  https://bitbucket.org/intelcilkruntime/itnel-cilk-runtime.git are
 *  not tracked.
 *  
 *  We welcome your contributions to this open source project. Thank you
 *  for your assistance in helping us improve Cilk Plus.
 **************************************************************************/

/**
 * @file cilk_fake.h
 *
 * @brief Macros to simulate a compiled Cilk program.
 *
 * Used carefully, these macros can be used to create a Cilk program with a
 * non-Cilk compiler by manually inserting the code necessary for interacting
 * with the Cilk runtime library.  They are not intended to be pretty (you
 * wouldn't want to write a whole program using these macros), but they are
 * useful for experiments.  They also work well as an illustration of what the
 * compiler generates.
 *
 * Details of the mechanisms used in these macros are described in
 * design-notes/CilkPlusABI.docx
 *
 * Example 1: fib in C++
 * ---------------------
 *
 *  #include <internal/cilk_fake.h>
 *  
 *  int fib(int n)
 *  {
 *      CILK_FAKE_PROLOG();
 *  
 *      if (n < 2)
 *          return n;
 *  
 *      int a, b;
 *      CILK_FAKE_SPAWN_R(a, fib(n - 1));
 *      b = fib(n - 2);
 *      CILK_FAKE_SYNC();
 *  
 *      return a + b;
 *  }
 *  
 *
 * Example 2: fib in C
 * -------------------
 *
 *  #include <internal/cilk_fake.h>
 *  
 *  int fib(int n);
 *  
 *  void fib_spawn_helper(__cilkrts_stack_frame* parent_sf, int* a, int n)
 *  {
 *      CILK_FAKE_SPAWN_HELPER_PROLOG(*parent_sf);
 *      *a = fib(n - 1);
 *      CILK_FAKE_SPAWN_HELPER_EPILOG();
 *  }
 *  
 *  int fib(int n)
 *  {
 *      CILK_FAKE_PROLOG();
 *  
 *      if (n < 2)
 *          return n;
 *  
 *      int a, b;
 *      CILK_FAKE_CALL_SPAWN_HELPER(fib_spawn_helper(&__cilk_sf, &a, n));
 *      b = fib(n - 2);
 *      CILK_FAKE_SYNC();
 *  
 *      CILK_FAKE_EPILOG();
 *      return a + b;
 *  }
 */

#ifndef INCLUDED_CILK_FAKE_DOT_H
#define INCLUDED_CILK_FAKE_DOT_H

// This header implements ABI version 1.  If __CILKRTS_ABI_VERSION is already
// defined but is less than 1, then the data structures in <internal/abi.h>
// will not match the expectations of facilities in this header.  Therefore,
// for successful compilation, __CILKRTS_ABI_VERSION must either be not
// defined, or defined to be 1 or greater.
#ifndef __CILKRTS_ABI_VERSION
    // ABI version was not specified.  Set it to 1.
#   define __CILKRTS_ABI_VERSION 1
#elif __CILKRTS_ABI_VERSION < 1
    // ABI version was specified but was too old.  Fail compilation.
#   error cilk_fake.h requirs an ABI version of 1 or greater
#endif

#include <internal/abi.h>

// alloca is defined in malloc.h on Windows, alloca.h on Linux
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
// Define offsetof
#include <stddef.h>
#endif

// Allows use of a different version that the one defined in abi.h
#define CILK_FAKE_VERSION_FLAG (__CILKRTS_ABI_VERSION << 24)
    
/* Initialize frame. To be called when worker is known */
__CILKRTS_INLINE void __cilk_fake_enter_frame_fast(__cilkrts_stack_frame *sf,
                                                   __cilkrts_worker      *w)
{
    sf->call_parent = w->current_stack_frame;
    sf->worker      = w;
    sf->flags       = CILK_FAKE_VERSION_FLAG;
    w->current_stack_frame = sf;
}

/* Initialize frame. To be called when worker is not known */
__CILKRTS_INLINE void __cilk_fake_enter_frame(__cilkrts_stack_frame *sf)
{
    __cilkrts_worker* w = __cilkrts_get_tls_worker();
    uint32_t          last_flag = 0;
    if (! w) {
        w = __cilkrts_bind_thread_1();
        last_flag = CILK_FRAME_LAST;
    }
    __cilk_fake_enter_frame_fast(sf, w);
    sf->flags |= last_flag;
}

/* Initialize frame. To be called within the spawn helper */
__CILKRTS_INLINE void __cilk_fake_helper_enter_frame(
    __cilkrts_stack_frame *sf,
    __cilkrts_stack_frame *parent_sf)
{
    sf->worker      = 0;
    sf->call_parent = parent_sf;
}

/* Called from the spawn helper to push the parent continuation on the task
 * deque so that it can be stolen.
 */
__CILKRTS_INLINE void __cilk_fake_detach(__cilkrts_stack_frame *sf)
{
    /* Initialize spawn helper frame.
     * call_parent was saved in __cilk_fake_helper_enter_frame */
    __cilkrts_stack_frame *parent = sf->call_parent;
    __cilkrts_worker *w = parent->worker;
    __cilk_fake_enter_frame_fast(sf, w);

    /* Append a node to the pedigree */
    sf->spawn_helper_pedigree = w->pedigree;
    parent->parent_pedigree = w->pedigree;
    w->pedigree.rank = 0;
    w->pedigree.parent = &sf->spawn_helper_pedigree;

    /* Push parent onto the task deque */
    __cilkrts_stack_frame *volatile *tail = w->tail;
    *tail++ = sf->call_parent;
    /* The stores must be separated by a store fence (noop on x86)
     * or the second store is a release (st8.rel on Itanium)   */
    w->tail = tail;
    sf->flags |= CILK_FRAME_DETACHED;
}

/* This variable is used in CILK_FAKE_FORCE_FRAME_PTR(), below */
static int __cilk_fake_dummy = 8;

/* The following macro is used to force the compiler into generating a frame
 * pointer.  We never change the value of __cilk_fake_dummy, so the alloca()
 * is never called, but we need the 'if' statement and the __cilk_fake_dummy
 * variable so that the compiler does not attempt to optimize it away.
 */
#define CILK_FAKE_FORCE_FRAME_PTR(sf) do {                              \
    if (__builtin_expect(1 & __cilk_fake_dummy, 0))                     \
        (sf).worker = (__cilkrts_worker*) alloca(__cilk_fake_dummy);    \
} while (0)

#ifndef CILK_FAKE_NO_SHRINKWRAP
    /* "shrink-wrap" optimization enabled.  Do not initialize frame on entry,
     * except to clear worker pointer.  Instead, defer initialization until
     * the first spawn.
     */
#   define CILK_FAKE_INITIAL_ENTER_FRAME(sf) ((void) ((sf).worker = 0))
#   define CILK_FAKE_DEFERRED_ENTER_FRAME(sf) do {            \
        if (! (sf).worker) __cilk_fake_enter_frame(&(sf));    \
    } while (0)
#else
    /* "shrink-wrap" optimization disabled.  Initialize frame immediately on
     * entry.  Do not initialize frame on spawn.
     */
#   define CILK_FAKE_INITIAL_ENTER_FRAME(sf) \
        __cilk_fake_enter_frame(&(sf))
#   define CILK_FAKE_DEFERRED_ENTER_FRAME(sf) ((void) &(sf))
#endif

/* Prologue of a spawning function.  Declares and initializes the stack
 * frame.
 */
#define CILK_FAKE_PROLOG()                                           \
    __cilk_fake_stack_frame __cilk_sf;                               \
    CILK_FAKE_FORCE_FRAME_PTR(__cilk_sf);                            \
    CILK_FAKE_INITIAL_ENTER_FRAME(__cilk_sf)

/* Prologue of a spawning function where the current worker is already known.
 * Declares and initializes the stack frame without looking up the worker from
 * TLS.
 */
#define CILK_FAKE_PROLOG_FAST(w)                                     \
    __cilk_fake_stack_frame __cilk_sf;                               \
    CILK_FAKE_FORCE_FRAME_PTR(__cilk_sf);                            \
    __cilk_fake_enter_frame_fast(&__cilk_sf, (w))

/* Simulate a cilk_sync */
#define CILK_FAKE_SYNC() CILK_FAKE_SYNC_IMP(__cilk_sf)

/* Epilog at the end of a spawning function.  Does a sync and calls the
 * runtime for leaving the frame.
 */
#ifdef __cplusplus
    // Epilogue is run automatically by __cilk_fake_stack_frame destructor.
#   define CILK_FAKE_EPILOG() ((void) __cilk_sf)
#else
#   define CILK_FAKE_EPILOG() CILK_FAKE_CLEANUP_FRAME(__cilk_sf)
#endif // C

/* Implementation of spawning function epilog.  See CILK_FAKE_EPILOG macro and
 * __cilk_fake_stack_frame destructor body.
 */
#define CILK_FAKE_CLEANUP_FRAME(sf) do {                     \
    if (! (sf).worker) break;                                \
    CILK_FAKE_SYNC_IMP(sf);                                  \
    CILK_FAKE_POP_FRAME(sf);                                 \
    if ((sf).flags != CILK_FAKE_VERSION_FLAG)                \
        __cilkrts_leave_frame(&(sf));                        \
} while (0)

/* Implementation of CILK_FAKE_SYNC with sf argument */
#define CILK_FAKE_SYNC_IMP(sf) do {                                       \
    if (__builtin_expect((sf).flags & CILK_FRAME_UNSYNCHED, 0))      {    \
        (sf).parent_pedigree = (sf).worker->pedigree;                     \
        CILK_FAKE_SAVE_FP(sf);                                            \
        if (! CILK_SETJMP((sf).ctx))                                      \
            __cilkrts_sync(&(sf));                                        \
    }                                                                     \
    ++(sf).worker->pedigree.rank;                                         \
} while (0)

/* Save the floating-point control registers.
 * The definition of CILK_FAKE_SAVE_FP is compiler specific (and
 * architecture specific on Windows)
 */
#ifdef _MSC_VER
#   define MXCSR_OFFSET offsetof(struct __cilkrts_stack_frame, mxcsr)
#   define FPCSR_OFFSET offsetof(struct __cilkrts_stack_frame, fpcsr)
#   if defined(_M_IX86)
/* Windows x86 */
#       define CILK_FAKE_SAVE_FP(sf) do {                               \
            __asm                                                       \
            {                                                           \
                mov eax, sf                                             \
                stmxcsr [eax+MXCSR_OFFSET]                              \
                fnstcw  [eax+FPCSR_OFFSET]                              \
            }                                                           \
        } while (0)
#   elif defined(_M_X64)
/* Windows Intel64 - Not needed - saved by setjmp call */
#       define CILK_FAKE_SAVE_FP(sf) ((void) sf)
#   else
#       error "Unknown architecture"
#   endif /* Microsoft architecture specifics */
#else
/* Non-Windows */
#   define CILK_FAKE_SAVE_FP(sf) do {                                   \
        __asm__ ( "stmxcsr %0\n\t"                                      \
                  "fnstcw %1" : : "m" ((sf).mxcsr), "m" ((sf).fpcsr));  \
    } while (0)
#endif

/* Call the spawn helper as part of a fake spawn */
#define CILK_FAKE_CALL_SPAWN_HELPER(helper) do {                    \
    CILK_FAKE_DEFERRED_ENTER_FRAME(__cilk_sf);                      \
    CILK_FAKE_SAVE_FP(__cilk_sf);                                   \
    if (__builtin_expect(! CILK_SETJMP(__cilk_sf.ctx), 1)) {        \
        helper;                                                     \
    }                                                               \
} while (0)

/* Body of a spawn helper function.  In addition to the worker and the
 * expression to spawn, pass it any number of statements to be executed before
 * detaching.
 */
#define CILK_FAKE_SPAWN_HELPER_BODY(parent_sf, expr, ...)                   \
    CILK_FAKE_SPAWN_HELPER_PROLOG(parent_sf);                               \
    __VA_ARGS__;                                                            \
    __cilk_fake_detach(&__cilk_sf);                                         \
    expr;                                                                   \
    CILK_FAKE_SPAWN_HELPER_EPILOG()

/* Prolog for a spawn helper function */
#define CILK_FAKE_SPAWN_HELPER_PROLOG(parent_sf)                     \
    __cilk_fake_spawn_helper_stack_frame __cilk_sf;                  \
    __cilk_fake_helper_enter_frame(&__cilk_sf, &(parent_sf))

/* Implementation of spawn helper epilog.  See CILK_FAKE_SPAWN_HELPER_EPILOG
 * and the __cilk_fake_spawn_helper_frame destructor.
 */
#define CILK_FAKE_SPAWN_HELPER_CLEANUP_FRAME(sf) do {                \
    if (! (sf).worker) break;                                        \
    CILK_FAKE_POP_FRAME(sf);                                         \
    __cilkrts_leave_frame(&(sf));                                    \
} while (0)

/* Epilog to execute at the end of a spawn helper */
#ifdef __cplusplus
    // Epilog handled by __cilk_fake_spawn_helper_stack_frame destructor
#   define CILK_FAKE_SPAWN_HELPER_EPILOG() ((void) __cilk_sf)
#else
#   define CILK_FAKE_SPAWN_HELPER_EPILOG() \
        CILK_FAKE_SPAWN_HELPER_CLEANUP_FRAME(__cilk_sf)
#endif

/* Pop the current frame off of the call chain */
#define CILK_FAKE_POP_FRAME(sf) do {                       \
    (sf).worker->current_stack_frame = (sf).call_parent;   \
    (sf).call_parent = 0;                                  \
} while (0)

#ifdef _WIN32
/* define macros for synching functions before allowing them to propagate. */
#   define CILK_FAKE_EXCEPT_BEGIN                              \
    if (0 == CILK_SETJMP(__cilk_sf.except_ctx)) {

#   define CILK_FAKE_EXCEPT_END                                               \
    } else {                                                                  \
        assert((__cilk_sf.flags & (CILK_FRAME_UNSYNCHED|CILK_FRAME_EXCEPTING))\
                == CILK_FRAME_EXCEPTING);                                     \
        __cilkrts_rethrow(&__cilk_sf);                                        \
        exit(0);                                                              \
    }
#else
#   define CILK_EXCEPT_BEGIN {
#   define CILK_EXCEPT_END   }
#endif

#ifdef __cplusplus
// The following definitions depend on C++ features.

// Wrap a functor (probably a lambda), so that a call to it cannot be
// inlined.
template <typename F>
class __cilk_fake_noinline_wrapper
{
    F&& m_fn;
public:
    __cilk_fake_noinline_wrapper(F&& fn) : m_fn(static_cast<F&&>(fn)) { }

#ifdef _WIN32
    __declspec(noinline) void operator()(__cilkrts_stack_frame *sf);
#else
    void operator()(__cilkrts_stack_frame *sf) __attribute__((noinline));
#endif

};

template <typename F>
void __cilk_fake_noinline_wrapper<F>::operator()(__cilkrts_stack_frame *sf)
{
    m_fn(sf);
}

template <typename F>
inline
__cilk_fake_noinline_wrapper<F> __cilk_fake_make_noinline_wrapper(F&& fn)
{
    return __cilk_fake_noinline_wrapper<F>(static_cast<F&&>(fn));
}

// Simulate "_Cilk_spawn expr", where expr must be a function call.
//
// Note: this macro does not correctly construct function arguments.
// According to the ABI specification, function arguments should be evaluated
// before the detach and destroyed after the detach.  This macro both
// evaluates and destroys them after the detach.  This means that if any part
// of the function argument expression depends on a value that is modified in
// the continuation of the spawn, race will occur between the continuation and
// the argument evaluation.
//
// To work around this problem, this macro accepts an arbitrary list of
// declarations and statements (separated by semicolons) that are evaluated
// before the detach.  Thus, to simulate:
//
//    _Cilk_spawn f(expr);
//
// one would write:
//
//    CILK_FAKE_SPAWN(f(arg), auto arg = expr);
//
// Despite appearing in the reverse order, the 'arg' variable is created and
// initialized before the detach and the call to f(arg) occurs after the
// detach.
#define CILK_FAKE_SPAWN(expr, ...)                                  \
    CILK_FAKE_CALL_SPAWN_HELPER(                                    \
        CILK_FAKE_SPAWN_HELPER(expr, __VA_ARGS__)(&__cilk_sf))

// Simulate "ret = cilk_spawn expr".  See CILK_FAKE_SPAWN for constraints.
#define CILK_FAKE_SPAWN_R(ret, expr, ...) \
    CILK_FAKE_SPAWN(((ret) = (expr)), __VA_ARGS__)

// Create a spawn helper as a C++11 lambda function.  In addition to the
// expression to spawn, this macro takes a any number of statements to be
// executed before detaching.
#define CILK_FAKE_SPAWN_HELPER(expr, ...)                                     \
    __cilk_fake_make_noinline_wrapper([&](__cilkrts_stack_frame *parent_sf) { \
        CILK_FAKE_SPAWN_HELPER_BODY(*parent_sf, expr, __VA_ARGS__);           \
    })

// C++ version of a __cilkrts_stack_frame for a spawning function.
// This struct is identical to __cilkrts_stack_frame except that the
// destructor automatically does frame cleanup.
struct __cilk_fake_stack_frame : __cilkrts_stack_frame
{
    // Extension of __cilkrts_stack_frame with constructor and destructor
    __cilk_fake_stack_frame() { }
    __forceinline ~__cilk_fake_stack_frame() {
        CILK_FAKE_CLEANUP_FRAME(*this);
    }
};

// C++ version of a __cilkrts_stack_frame for a spawn helper.
// This struct is identical to __cilkrts_stack_frame except that the
// destructor automatically does frame cleanup.
struct __cilk_fake_spawn_helper_stack_frame : __cilkrts_stack_frame
{
    // Extension of __cilkrts_stack_frame with constructor and destructor
    __cilk_fake_spawn_helper_stack_frame() { worker = 0; }
    __forceinline ~__cilk_fake_spawn_helper_stack_frame() {
        CILK_FAKE_SPAWN_HELPER_CLEANUP_FRAME(*this);            
    }
};
#else
// For C, __cilk_fake_stack_frame and __cilk_fake_spawn_helper_stack_frame are
// identical to __cilkrts_stack_frame.  Frame cleanup must be performed
// excplicitly (in CILK_FAKE_EPILOG and CILK_FAKE_SPAWN_HELPER_EPILOG)
typedef __cilkrts_stack_frame __cilk_fake_stack_frame;
typedef __cilkrts_stack_frame __cilk_fake_spawn_helper_stack_frame;
#endif

#endif // ! defined(INCLUDED_CILK_FAKE_DOT_H)
