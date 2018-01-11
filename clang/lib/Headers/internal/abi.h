/*
 *  abi.h
 *
 *  Copyright (C) 2009-2016, Intel Corporation
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
 *
 ******************************************************************************/

/**
 * @file abi.h
 *
 * @brief Defines the application binary interface between the compiler and
 * the Intel Cilk Plus runtime.
 */

#ifndef CILK_INTERNAL_ABI_H
#define CILK_INTERNAL_ABI_H


#include <cilk/common.h>
#include <stddef.h>  // Needed to define size_t

/**
 * Jump buffers are OS and architecture dependent
 */
#if ! defined(_MSC_VER)
/* Non-Windows - only need 5 registers for the jump buffer for both IA32 and Intel64 */
typedef void *__CILK_JUMP_BUFFER[5];

/** OS-specific implementation of setjmp */
#   define CILK_SETJMP(X) __builtin_setjmp(X)
/** OS-specific implementation of longjmp */
#   define CILK_LONGJMP(X) __builtin_longjmp(X,1)
#else
/* Windows - things are a little more complicated */
#   if defined(_M_X64)
/* Intel64 - Use an OS-defined jump buffer */
#       include <setjmp.h>
typedef jmp_buf __CILK_JUMP_BUFFER;

#       define CILK_SETJMP(X) setjmp(X)
#       define CILK_LONGJMP(X) longjmp(X, 1)
#   elif defined(_M_IX86)
/**
 * Windows x86 - Use a simplified version of the Windows jump buffer for x86
 * setjmp is provided by __cilkrts_setjmp which passes jump buffer in EAX and
 * destination in EDX longjmp is provided by an internal routine which uses
 * this structure
 */
typedef struct
{
    unsigned long Ebp;
    unsigned long Ebx;
    unsigned long Edi;
    unsigned long Esi;
    unsigned long Esp;
    unsigned long Eip;
    unsigned long Registration;
    unsigned long TryLevel;
} __CILK_JUMP_BUFFER;

#    else
#    error Unexpected architecture - Need to define __CILK_JUMP_BUFFER
#    endif  /* _M_X64 */

#endif  /* defined(_MSC_VER) */

/* struct tags */
typedef struct __cilkrts_stack_frame __cilkrts_stack_frame; ///< struct tag for stack frame

// Forwarded declarations
typedef struct global_state_t        global_state_t;  ///< Forwarded declaration for global state
typedef struct local_state           local_state;     ///< Forwarded declaration for local state
typedef struct cilkred_map           cilkred_map;     ///< Forward declaration for reducer map

/// Forwarded declaration for system-dependent worker state
typedef struct __cilkrts_worker_sysdep_state
                                     __cilkrts_worker_sysdep_state;

/**
 * The worker struct contains per-worker information that needs to be
 * visible to the compiler, or rooted here.
 *
 * For 32-bit Windows we need to be aligning the structures on 4-byte
 * boundaries to match where ICL is allocating the birthrank and rank
 * in the __cilkrts_stack_frame.  It's 4-byte aligned instead of 8-byte
 * aligned.  This is OK because the compiler is dealing with the 64-bit
 * quantities as two 32-bit values.  So change the packing to be on
 * 4-byte boundaries.
 *
 * The fields of the worker struct can be classified as either local
 * or shared.
 *
 *  Local: This field is only accessed by the thread bound to this
 *    worker struct.  Local fields can be freely accessed without
 *    acquiring locks.
 *  
 *  Shared: This field may be accessed by multiple worker threads.
 *    Accesses to shared fields usually requires locks, except in
 *    special situations where one can prove that locks are
 *    unnecessary.
 *
 * The fields of the worker struct can also be classified as
 * "read-only" if the field does not change after it is initialized.
 * Otherwise, the field is "read/write".  Read-only fields do not
 * require locks to access (ignoring the synchronization that might be
 * needed for initialization if this can occur in parallel).
 *
 * Finally, we explicitly classify some fields as "synchronization"
 * fields if they are used as part of a synchronization protocol in
 * the runtime.  These variables are generally shared and read/write.
 * Mostly, this category includes lock variables and other variables
 * that are involved in synchronization protocols (i.e., the THE
 * protocol).
 */
#if defined(_MSC_VER) && defined(_M_IX86)
#pragma pack(push, 4)
#endif

struct __cilkrts_worker {
    /**
     * T, H, and E pointers in the THE protocol See "The implementation of
     * the Cilk-5 multithreaded language", PLDI 1998:
     * http://portal.acm.org/citation.cfm?doid=277652.277725
     *
     * Synchronization fields.  [shared read/write]
     */
    __cilkrts_stack_frame *volatile *volatile tail;
    __cilkrts_stack_frame *volatile *volatile head;  /**< @copydoc tail */
    __cilkrts_stack_frame *volatile *volatile exc;   /**< @copydoc tail */

    /**
     * Addition to the THE protocol to allow us to protect some set of
     * entries in the tail queue from stealing.  Normally, this is set
     * beyond the end of the task queue, indicating that all entries are
     * available for stealing.  During exception handling, protected_tail
     * may be set to the first entry in the task queue, indicating that
     * stealing is not allowed.
     *
     * Synchronization field.
     */
    __cilkrts_stack_frame *volatile *volatile protected_tail;

    /**
     * Limit of the Lazy Task Queue, to detect queue overflow
     * [local read-only]
     */
    __cilkrts_stack_frame *volatile *ltq_limit;

    /**
     * Worker id.
     * [local read-only]
     */
    int32_t self;

    /**
     * Global state of the runtime system, opaque to the client.
     * [local read-only]
     */
    global_state_t *g;

    /**
     * Additional per-worker state of the runtime system that we want
     * to maintain hidden from the client.
     * [shared read-only]
     */
    local_state *l;

    /**
     * Map from reducer names to reducer values.
     * [local read/write]
     */
    cilkred_map *reducer_map;

    /**
     * A slot that points to the currently executing Cilk frame.
     * [local read/write]
     */
    __cilkrts_stack_frame *current_stack_frame;

    /**
     * Reserved space for a pointer.
     * Used to be __cilkrts_stack_frame *volatile *volatile saved_protected_tail; 
     */
    void* reserved;

    /**
     * System-dependent part of the worker state
     * [local read-only]
     */
    __cilkrts_worker_sysdep_state *sysdep;

#if __CILKRTS_ABI_VERSION >= 1
    /**
     * Per-worker pedigree information used to support scheduling-independent
     * pseudo-random numbers.
     * [local read/write]
     */
    __cilkrts_pedigree   pedigree;    
#endif  /* __CILKRTS_ABI_VERSION >= 1 */
};


/**
 * Every spawning function has a frame descriptor.  A spawning function
 * is a function that spawns or detaches.  Only spawning functions
 * are visible to the Cilk runtime.
 */
struct __cilkrts_stack_frame
{
    /**
     * flags is an integer with values defined below.  Client code
     * initializes flags to CILK_FRAME_VERSION before the first Cilk
     * operation.
     *
     * The low 24-bits of the 'flags' field are the flags, proper.  The high
     * 8-bits are the version number.
     *
     * IMPORTANT: bits in this word are set and read by the PARENT ONLY,
     * not by a spawned child.  In particular, the STOLEN and UNSYNCHED
     * bits are set on a steal and are read before a sync.  Since there
     * is no synchronization (locking) on this word, any attempt to set
     * or read these bits asynchronously in a child would result in a race.
     */
    uint32_t flags;

    /** Not currently used.  Not initialized by Intel compiler. */
    int32_t size;

    /** 
     * call_parent points to the __cilkrts_stack_frame of the closest
     * ancestor spawning function, including spawn helpers, of this frame.
     * It forms a linked list ending at the first stolen frame.
     */
    __cilkrts_stack_frame *call_parent;

    /**
     * The client copies the worker from TLS here when initializing
     * the structure.  The runtime ensures that the field always points
     * to the __cilkrts_worker which currently "owns" the frame.
     */
    __cilkrts_worker *worker;

    /**
     * Unix: Pending exception after sync.  The sync continuation
     * must call __cilkrts_rethrow to handle the pending exception.
     *
     * Windows: the handler that _would_ have been registered if our
     * handler were not there.  We maintain this for unwinding purposes.
     * Win32: the value of this field is only defined in spawn helper
     * functions
     *
     * Win64: except_data must be filled in  for all functions with a
     * __cilkrts_stack_frame
     */
    void *except_data;

    /**
     * Before every spawn and nontrivial sync the client function
     * saves its continuation here.
     */
    __CILK_JUMP_BUFFER ctx;

#if __CILKRTS_ABI_VERSION >= 1
    /**
     * Architecture-specific floating point state.  mxcsr and fpcsr should be
     * set when CILK_SETJMP is called in client code.  Note that the Win64
     * jmpbuf for the Intel64 architecture already contains this information
     * so there is no need to use these fields on that OS/architecture.
     */
    uint32_t mxcsr;
    uint16_t fpcsr;         /**< @copydoc mxcsr */


    /**
     * reserved is not used at this time.  Client code should initialize it
     * to 0 before the first Cilk operation
     */
    uint16_t reserved;

    /**
     * Pedigree information to support scheduling-independent pseudo-random
     * numbers.  There are two views of this information.  The copy in a
     * spawning function is used to stack the rank and communicate to the
     * runtime on a steal or continuation.  The copy in a spawn helper is
     * immutable once the function is detached and is a node in the pedigree.
     * The union is used to make clear which view we're using.
     *
     * In the detach sequence Client code should:
     *    - copy the worker pedigree into the spawn helper's pedigree
     *    - copy the worker pedigree into the call parent's pedigree
     *    - set the worker's rank to 0
     *    - set the worker's pedigree.next to the spawn helper's pedigree
     */
    union
    {
        __cilkrts_pedigree spawn_helper_pedigree; /* Used in spawn helpers */
        __cilkrts_pedigree parent_pedigree;       /* Used in spawning funcs */
    };
#endif  /* __CILKRTS_ABI_VERSION >= 1 */
};

/*
 * Restore previous structure packing for 32-bit Windows
 */
#if defined(_MSC_VER) && defined(_M_IX86)
#pragma pack(pop)
#endif

/* Values of the flags bitfield */
/** CILK_FRAME_STOLEN is set if the frame has ever been stolen. */
#define CILK_FRAME_STOLEN    0x01

/**
 * CILK_FRAME_UNSYNCHED is set if the frame has been stolen and
 * is has not yet executed _Cilk_sync.  It is technically a misnomer in that a
 * frame can have this flag set even if all children have returned.
 */
#define CILK_FRAME_UNSYNCHED 0x02

/**
 * Is this frame detached (spawned)?  If so the runtime needs
 * to undo-detach in the slow path epilogue.
 */
#define CILK_FRAME_DETACHED  0x04

/**
 * CILK_FRAME_EXCEPTION_PROBED is set if the frame has been probed in the
 * exception handler first pass
 */
#define CILK_FRAME_EXCEPTION_PROBED 0x08

/** Is this frame receiving an exception after sync? */
#define CILK_FRAME_EXCEPTING 0x10

/**
 * Is the pedigree unsynched?  That is, has a synch occurred that is not
 * yet represented in the pedigree?
 */
#define CILK_FRAME_SF_PEDIGREE_UNSYNCHED 0x20

/** Is this the last (oldest) Cilk frame? */
#define CILK_FRAME_LAST	     0x80

/**
 * Is this frame in the epilogue, or more generally after the last
 * sync when it can no longer do any Cilk operations?
 */
#define CILK_FRAME_EXITING   0x0100

/** Is this frame suspended? (used for debugging) */
#define CILK_FRAME_SUSPENDED 0x8000

/** Used by Windows exception handling to indicate that __cilkrts_leave_frame should do nothing */
#define CILK_FRAME_UNWINDING 0x10000

/*
 * The low 24-bits of the 'flags' field are the flags, proper.  The high 8-bits
 * are the version number.
 */

/** ABI version left shifted to the high byte */
#define CILK_FRAME_VERSION (__CILKRTS_ABI_VERSION << 24)

/** Mask for the flags field to isolate the version bits */
#define CILK_FRAME_VERSION_MASK  0xFF000000

/** Mask for the flags field to isolate the flag bits */
#define CILK_FRAME_FLAGS_MASK    0x00FFFFFF

/** Convenience macro to provide access the version portion of the flags field */
#define CILK_FRAME_VERSION_VALUE(_flags) (((_flags) & CILK_FRAME_VERSION_MASK) >> 24)

/** Any undefined bits are reserved and must be zero ("MBZ" = "Must Be Zero") */
#define CILK_FRAME_MBZ  (~ (CILK_FRAME_STOLEN | \
                            CILK_FRAME_UNSYNCHED | \
                            CILK_FRAME_DETACHED | \
                            CILK_FRAME_EXCEPTION_PROBED | \
                            CILK_FRAME_EXCEPTING | \
                            CILK_FRAME_SF_PEDIGREE_UNSYNCHED | \
                            CILK_FRAME_LAST | \
                            CILK_FRAME_EXITING | \
                            CILK_FRAME_SUSPENDED | \
                            CILK_FRAME_UNWINDING | \
                            CILK_FRAME_VERSION_MASK))

__CILKRTS_BEGIN_EXTERN_C

/**
 * Call __cilkrts_enter_frame to initialize an ABI 0 frame descriptor.
 * Initialize the frame descriptor before spawn or detach.  A function that
 * conditionally does Cilk operations need not initialize the frame descriptor
 * in a code path that never uses it.
 *
 * @param sf The __cilkrts_stack_frame that is to be initialized.
 */
CILK_ABI(void) __cilkrts_enter_frame(__cilkrts_stack_frame* sf);

/**
 * Call __cilkrts_enter_frame to initialize an ABI 1 frame descriptor.
 * Initialize the frame descriptor before spawn or detach.  A function that
 * conditionally does Cilk operations need not initialize the frame descriptor
 * in a code path that never uses it.
 *
 * @param sf The __cilkrts_stack_frame that is to be initialized.
 */
CILK_ABI(void) __cilkrts_enter_frame_1(__cilkrts_stack_frame* sf);

/**
 * __cilkrts_enter_frame_fast is the same as __cilkrts_enter_frame, except it
 * assumes that the thread has already been bound to a worker.
 *
 * @param sf The __cilkrts_stack_frame that is to be initialized.
 */
CILK_ABI(void) __cilkrts_enter_frame_fast(__cilkrts_stack_frame *sf);

/**
 * __cilkrts_enter_frame_fast_1 is the same as __cilkrts_enter_frame_1,
 * except it assumes that the thread has already been bound to a worker.
 *
 * @param sf The __cilkrts_stack_frame that is to be initialized.
 */
CILK_ABI(void) __cilkrts_enter_frame_fast_1(__cilkrts_stack_frame *sf);

/**
 * Call leave_frame before leaving a frame, after sync.  This function
 * returns except in a spawn wrapper where the parent has been stolen.
 *
 * @param sf The __cilkrts_stack_frame that is to be left.
 */
CILK_ABI(void) __cilkrts_leave_frame(__cilkrts_stack_frame *sf);

/**
 * Suspends the runtime by notifying the workers that they should not try to
 * steal. This function is supposed to be called from a non-parallel region
 * (i.e., after cilk_sync in the top-level spawning function). Otherwise,
 * which workers are sleeping or busy is unpredictable in general.
 * The runtime can be resumed by calling __cilkrts_resume().
 */
CILK_ABI(void) __cilkrts_suspend(void);

/**
 * Resumes the runtime by notifying the workers that they can steal.
 */
CILK_ABI(void) __cilkrts_resume(void);

/**
 * Wait for any spawned children of this function to complete before
 * continuing.  This function will only return when the join counter
 * has gone to 0.  Other workers will re-enter the scheduling loop to
 * attempt to steal additional work.
 *
 * @param sf The __cilkrts_stack_frame that is to be synched.
 */
CILK_ABI(void) __cilkrts_sync(__cilkrts_stack_frame *sf);

/**
 * Called when an exception is escaping a spawn * wrapper.
 * The stack frame's except_data field is the C++ runtime
 * exception object.  If NULL (temporary workaround) the
 * currently caught exception should be rethrown.  If this
 * function returns normal exit functions must be called;
 * undo-detach will have been done.
 *
 * @param sf The __cilkrts_stack_frame for the function that
 * is raising an exception.
 */
CILK_ABI_THROWS(void)
    __cilkrts_return_exception(__cilkrts_stack_frame *sf);

/**
 * Called to re-raise an exception.
 *
 * @param sf The __cilkrts_stack_frame for the function that
 * is raising an exception.
 */
CILK_ABI_THROWS(void) __cilkrts_rethrow(__cilkrts_stack_frame *sf);

/**
 * Called at the beginning of a spawning function to get the worker
 * that this function is running on.  This worker will be used to
 * initialize the __cilkrts_stack_frame.
 *
 * @return The __cilkrts_worker that the function is running on.
 * @return NULL if this thread is not yet bound to a worker.
 */
CILK_ABI(__cilkrts_worker_ptr) __cilkrts_get_tls_worker(void);

/**
 * Similar to __cilkrts_get_tls_worker, but assumes that TLS has been
 * initialized.
 *
 * @return The __cilkrts_worker that the function is running on.
 * @return NULL if this thread is not yet bound to a worker.
 */
CILK_ABI(__cilkrts_worker_ptr) __cilkrts_get_tls_worker_fast(void);

/**
 * Binds a thread to the runtime by associating a __cilkrts_worker with
 * it.  Called if __cilkrts_get_tls_worker returns NULL.  This function will
 * initialize the runtime the first time it is called.
 *
 * This function is versioned by the ABI version number.  The runtime
 * will export all previous versions.  This prevents using an application
 * built with a newer compiler against an old runtime.
 *
 * @return The __cilkrts_worker bound to the thread the function is running
 * on.
 */
CILK_ABI(__cilkrts_worker_ptr) __cilkrts_bind_thread_1(void);

typedef uint32_t cilk32_t;  /**< 32-bit unsigned type for cilk_for loop indicies */

typedef uint64_t cilk64_t;  /**< 64-bit unsigned type for cilk_for loop indicies */

/**
 * Signature for the lambda function generated for the body of a cilk_for loop
 * which uses 32-bit indicies
 */
typedef void (*__cilk_abi_f32_t)(void *data, cilk32_t low, cilk32_t high);

/**
 * Signature for the lambda function generated for the body of a cilk_for lop
 * which uses 64-bit indicies
 */
typedef void (*__cilk_abi_f64_t)(void *data, cilk64_t low, cilk64_t high);

/**
 * @brief cilk_for implementation for 32-bit indexes.
 *
 * @param body The lambda function for the body of the cilk_for.  The lambda
 * function will be called to execute each grain of work.
 * @param data Data passed by the compiler into the lambda function.  Provides
 * access to data outside the cilk_for body.
 * @param count Number of steps in the loop.
 * @param grain This parameter allows the compiler to pass a value from a
 * \#pragam(grainsize) statement to allow the user to control the grainsize.  If
 * there isn't a \#pragma(grainsize) immediately preceeding cilk_for loop, Pass
 * 0 to specify that the runtime should calculate the grainsize using its own
 * hueristicts.
 */
CILK_ABI_THROWS(void) __cilkrts_cilk_for_32(__cilk_abi_f32_t body,
                                            void *data,
                                            cilk32_t count,
                                            int grain);

/**
 * @brief cilk_for implementation for 64-bit indexes.
 *
 * @copydetails __cilkrts_cilk_for_32
 */
CILK_ABI_THROWS(void) __cilkrts_cilk_for_64(__cilk_abi_f64_t body,
                                            void *data,
                                            cilk64_t count,
                                            int grain);

/**
 * @brief Allocate memory for variable length arrays. If the frame is
 * sync'd, the memory will be allocated on the stack, otherwise it will
 * be allocated from the heap.
 *
 * @param sf The __cilkrts_stack_frame for the function allocating the
 * memory.
 * @param size The number of bytes requested.
 * @param distance_from_sp_to_alloca_area ?.
 * @param align Alignment required.  Always >= minimum stack alignment,
 * >= ptr_size, and always a power of 2.
 * @param needs_tag Non-zero if the pointer being returned needs to be
 * tagged
 *
 * @return The address of the memory block allocated.
 */

CILK_ABI(__cilkrts_void_ptr)
__cilkrts_stack_alloc(__cilkrts_stack_frame *sf,
                      size_t size,
                      size_t distance_from_sp_to_alloca_area,
                      uint32_t align,
                      uint32_t needs_tag);

/**
 * @brief Free memory allocated by _cilkrts_stack_alloc() for variable length
 * arrays.
 *
 * @param sf The __cilkrts_stack_frame for the function allocating the
 * memory.
 * @param p Pointer to the memory block to be freed.
 * @param size The number of bytes requested.
 * @param distance_from_sp_to_alloca_area ?.
 * @param align Alignment required.  Always >= minimum stack alignment,
 * >= ptr_size, and always a power of 2.
 * @param know_from_stack Non-zero if the pointer is known to have been
 * allocated on the stack and has no tag.
 */
CILK_ABI(void)
__cilkrts_stack_free(__cilkrts_stack_frame *sf,
                     void *p,
                     size_t size,
                     size_t distance_from_sp_to_alloca_area,
                     uint32_t align,
                     uint32_t known_from_stack);

/**
 * @brief System-dependent code to save floating point control information
 * to an ABI 1 or higher @c __cilkrts_stack_frame.  If possible (and necessary)
 * the code to save the floating point control information should be inlined.
 *
 * Note that this function does *not* save the current floating point
 * registers.  It saves the floating point control words that control
 * precision and rounding and stuff like that.
 *
 * This function will be a noop for architectures that don't have warts
 * like the floating point control words, or where the information is
 * already being saved by the setjmp.
 *
 * @param sf  @c __cilkrts_stack_frame for the frame we're saving the
 * floating point control information in.
 */
CILK_ABI(void)
__cilkrts_save_fp_ctrl_state(__cilkrts_stack_frame *sf);

__CILKRTS_END_EXTERN_C
#endif /* include guard */
