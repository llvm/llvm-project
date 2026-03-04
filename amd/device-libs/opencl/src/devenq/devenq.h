
#include "oclc.h"
#include "device_amd_hsa.h"

#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

//! AmdAqlWrap slot state
enum AqlWrapState {
    AQL_WRAP_FREE = 0,
    AQL_WRAP_RESERVED,
    AQL_WRAP_READY,
    AQL_WRAP_MARKER,
    AQL_WRAP_BUSY,
    AQL_WRAP_DONE
};

//! Profiling states
enum ProfilingState {
    PROFILING_COMMAND_START = 0,
    PROFILING_COMMAND_END,
    PROFILING_COMMAND_COMPLETE
};

typedef struct _AmdVQueueHeader {
    uint    aql_slot_num;       //!< [LRO/SRO] The total number of the AQL slots (multiple of 64).
    uint    event_slot_num;     //!< [LRO] The number of kernel events in the events buffer
    ulong   event_slot_mask;    //!< [LRO] A pointer to the allocation bitmask array for the events
    ulong   event_slots;        //!< [LRO] Pointer to a buffer for the events.
                                // Array of event_slot_num entries of AmdEvent
    ulong   aql_slot_mask;      //!< [LRO/SRO]A pointer to the allocation bitmask for aql_warp slots
    uint    command_counter;    //!< [LRW] The global counter for the submitted commands into the queue
    uint    wait_size;          //!< [LRO] The wait list size (in clk_event_t)
    uint    arg_size;           //!< [LRO] The size of argument buffer (in bytes)
    uint    mask_groups;        //!< [LRO] The mask group size
    ulong   kernel_table;       //!< [LRO] Pointer to an array with all kernel objects (ulong for each entry)
    uint    reserved[2];        //!< For the future usage
} AmdVQueueHeader;

struct _AmdEvent;

typedef struct _AmdAqlWrap {
    uint state;             //!< [LRW/SRW] The current state of the AQL wrapper:  FREE, RESERVED, READY,
                            // MARKER, BUSY and DONE. The block could be returned back to a free state.
    uint enqueue_flags;     //!< [LWO/SRO] Contains the flags for the kernel execution start -
                            //  (KERNEL_ENQUEUE_FLAGS_T)
                            // NO_WAIT - we just start processing
                            // WAIT_PARENT - check if parent_wrap->state is done and then start processing
                            // WAIT_WORK_GROUP currently == WAIT_PARENT
    uint command_id;        //!< [LWO/SRO] The unique command ID
    uint child_counter;     //!< [LRW/SRW] Counter that determine the launches of child kernels.
                            // It's incremented on the
                            // start and decremented on the finish. The parent kernel can be considered as
                            // done when the value is 0 and the state is DONE

    //!< [LWO/SRO] CL event for the current execution (clk_event_t)
    union {
        __global struct _AmdEvent *completion;
        ulong completion_padding;
    };

    //!< [LWO/SRO] Pointer to the parent AQL wrapper (AmdAqlWrap*)
    union {
        __global struct _AmdAqlWrap *parent_wrap;
        ulong parent_padding;
    };

    union {
        __global size_t *wait_list;  //!< [LRO/SRO] Pointer to an array of clk_event_t objects (64 bytes default)
        ulong wait_list_padding;
    };

    uint wait_num;          //!<  [LWO/SRO] The number of cl_event_wait objects
    uint reserved[5];       //!< For the future usage
    hsa_kernel_dispatch_packet_t aql;  //!< [LWO/SRO] AQL packet - 64 bytes AQL packet
} AmdAqlWrap;

typedef struct _AmdEvent {
    uint state;             //!< [LRO/SRW] Event state: START, END, COMPLETE
    uint counter;           //!< [LRW] Event retain/release counter. 0 means the event is free
    ulong timer[3];         //!< [LRO/SWO] Timer values for profiling for each state
    ulong capture_info;     //!< [LRW/SRO] Profiling capture info for CLK_PROFILING_COMMAND_EXEC_TIME
} AmdEvent;

// XXX this needs to match workgroup/wg.h MAX_WAVES_PER_SIMD
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 256

// ABI has implicit trailing arguments
#define NUM_IMPLICIT_ARGS (__oclc_ABI_version < 500 ? 7 : 32)

static inline __global void *
get_printf_ptr(void)
{
    if (__oclc_ABI_version < 500) {
        return (__global void *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[3]);
    } else {
        return (__global void *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[9]);
    }
}

static inline __global AmdVQueueHeader *
get_vqueue(void)
{
    if (__oclc_ABI_version < 500) {
        return (__global AmdVQueueHeader *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[4]);
    } else {
        return (__global AmdVQueueHeader *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[13]);
    }
}

static inline __global AmdAqlWrap *
get_aql_wrap(void)
{
    if (__oclc_ABI_version < 500) {
        return (__global AmdAqlWrap *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[5]);
    } else {
        return (__global AmdAqlWrap *)(((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[14]);
    }
}

static inline size_t
get_bases(void)
{
    return ((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[24];
}

static inline size_t
get_hsa_queue(void)
{
    return ((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[25];
}

// reserve a slot in a bitmask controlled resource
// n is the number of slots
static inline int
reserve_slot(__global uint * restrict mask, uint n, uint mask_groups)
{
    n >>= 5;
    uint j, k, v, vv, z;

    // Spread the starting points
    k = (get_local_linear_id() * mask_groups) % n;

    // Make only one pass
    for (j=0;j<n;++j) {
        __global atomic_uint *p = (__global atomic_uint *)(mask + k);
        v = atomic_load_explicit(p, memory_order_relaxed, memory_scope_device);
        for (;;) {
            z = ctz(~v);
            if (z == 32U)
                break;
            vv = v | (1U << z);
            if (atomic_compare_exchange_strong_explicit(p, &v, vv, memory_order_relaxed, memory_order_relaxed, memory_scope_device))
                break;
        }
        if (z < 32U)
            break;
        k = k == n-1 ? 0 : k+1;
    }

    k = (k << 5) + z;
    return z < 32U ? (int)k : -1;
}

// release slot in a bitmask controlled resource
// i is the slot number
static inline void
release_slot(__global uint * restrict mask, uint i)
{
    /* uint b = ~(1UL << (i & 0x1f)); */
    // FIXME: Use llvm.ptrmask
    uint b = ~amd_bfm(1U, i);
    __global atomic_uint *p = (__global atomic_uint *)(mask + (i >> 5));
    uint v, vv;

    v = atomic_load_explicit(p, memory_order_relaxed, memory_scope_device);
    for (;;) {
        vv = v & b;
        if (atomic_compare_exchange_strong_explicit(p, &v, vv, memory_order_relaxed, memory_order_relaxed, memory_scope_device))
            break;
    }
}

static inline uint
align_up(uint start, uint align)
{
    return (start + align - 1U) & -align;
}

