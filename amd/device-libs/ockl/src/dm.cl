/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "oclc.h"
#include "ockl.h"

#define AS_UINT2(X) __builtin_astype(X, uint2)
#define AS_ULONG(X) __builtin_astype(X, ulong)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

extern ulong __ockl_devmem_request(ulong addr, ulong size);

// Define this to track user requested non-slab (i.e. "large") in-use
// allocations. This adds the definition of a query function nna() that
// returns a snapshot of the current value.
#define NON_SLAB_TRACKING 1

// The number of kinds of blocks.  Do not change.
#define NUM_KINDS 16

// The size where we switch the large & slow mechanism.  Do not change.
#define ALLOC_THRESHOLD 3072

// This controls the size of the heap, and also how often
// we need to expand the capacity of the array that tracks
// the allocations that have been made.
//
// With the definition below, 256, one level can hold 256
// slabs (512 MiB), and two levels can hold (256+1)*256 = 65792
// slabs (131585 MiB)
#define SDATA_SHIFT 8
#define NUM_SDATA (1 << SDATA_SHIFT)
#define SDATA_MASK (NUM_SDATA - 1)
#define MAX_RECORDABLE_SLABS ((NUM_SDATA + 1) * NUM_SDATA)

// Type of variable use to hold a kind
typedef uint kind_t;

// Type of variable used to hold a sdata index
typedef uint sid_t;

// Type of variable used to hold a group mask
typedef ulong gmask_t;

// Various info about a given kind of block
struct kind_info_s {
    uint num_blocks;
    uint num_usable_blocks;
    uint skip_threshold;
    uint block_offset;
    uint first_unusable;
    uint gap_unusable;
    uint pattern_unusable;
};

static const __constant struct kind_info_s kinfo[NUM_KINDS] = {
    { /*  0:   16 */ 130054, 129546, 110114, 16288,    6, 256, 0x00000000 },
    { /*  1:   24 */  86927,  86758,  73744, 10904,  399, 512, 0x00000000 },
    { /*  2:   32 */  65280,  64770,  55054,  8192,    0, 128, 0x00000000 },
    { /*  3:   48 */  43576,  43406,  36895,  5504,   56, 256, 0x00000000 },
    { /*  4:   64 */  32703,  32193,  27364,  4160,   63,  64, 0x00000000 },
    { /*  5:   96 */  21816,  21646,  18399,  2816,   56, 128, 0x00000000 },
    { /*  6:  128 */  16367,  15856,  13477,  2176,   15,  32, 0x00008000 },
    { /*  7:  192 */  10915,  10745,   9133,  1472,   35,  64, 0x00000000 },
    { /*  8:  256 */   8187,   7676,   6524,  1280,   11,  16, 0x08000800 },
    { /*  9:  384 */   5459,   5289,   4495,   896,   19,  32, 0x00080000 },
    { /* 10:  512 */   4094,   3583,   3045,  1024,    6,   8, 0x40404040 },
    { /* 11:  768 */   2730,   2560,   2176,   512,   10,  16, 0x04000400 },
    { /* 12: 1024 */   2047,   1536,   1305,  1024,    3,   4, 0x88888888 },
    { /* 13: 1536 */   1365,   1195,   1015,   512,    5,   8, 0x20202020 },
    { /* 14: 2048 */   1023,    512,    435,  2048,    1,   2, 0xaaaaaaaa },
    { /* 15: 3072 */    682,    512,    435,  2048,    2,   4, 0x44444444 },
};

// A slab is a chunk of memory used to provide "block"s whose addresses are
// returned by malloc.  The slab tracks which blocks are in use using a bit
// array "bits".  The blocks themselves start at offset "block_offset".
typedef struct slab_s {
    kind_t k;            // The kind of the blocks
    sid_t i;             // The index of the slab in the heap
    atomic_uint start;   // Used to guide the search for unused blocks
    uint pad;
    atomic_uint in_use[2*1024*1024 / 4 - 4];  // An array of per-block bits, followed by the blocks
} slab_t;

// The minimum number of ticks each slab allocation must be separated by
#define SLAB_TICKS 20000

// This struct captures a little more information about a given slab
// such as its address and its number of used blocks.  There is another
// member used to increase the number of slabs that can be recorded in
// the heap
typedef struct sdata_s {
    atomic_ulong array;               // Address of an array of sdata_t
    atomic_ulong saddr;               // Slab address is really a __global slab_t *
    atomic_uint num_used_blocks;
} sdata_t;

// The number of ulong that cover an sdata_t
#define ULONG_PER_SDATA 3

// The length of a CAS loop sleep
#define CAS_SLEEP 2

// This is used to communicate that a result is
// not currently available due to a limit on how
// fast we are allowed to create new slabs
#define SDATA_BUSY (__global sdata_t *)1

// Possible results when trying to increase the number of recordable slabs
#define GROW_SUCCESS 0
#define GROW_BUSY 1
#define GROW_FAILURE 2

// The minimum number of ticks each grow must be separated by
#define GROW_TICKS 30000

// The number of ulong per cache line used to separate atomics
#define ULONG_PER_CACHE_LINE 16
#define ATOMIC_PAD (ULONG_PER_CACHE_LINE-1)

// Type used to hold a search start index
typedef struct start_s {
    atomic_uint value;
#if ATOMIC_PAD > 0
    ulong pad[ATOMIC_PAD];
#endif
} start_t;

// Type used to hold the number of allocated slabs
typedef struct nallocated_s {
    atomic_uint value;
#if ATOMIC_PAD > 0
    ulong pad[ATOMIC_PAD];
#endif
} nallocated_t;

// Type used to hold the number of recordable slabs
typedef struct nrecordable_s {
    atomic_uint value;
#if ATOMIC_PAD > 0
    ulong pad[ATOMIC_PAD];
#endif
} nrecordable_t;

// Type used to hold a real-time clock sample
typedef struct rtcsample_s {
    atomic_ulong value;
#if ATOMIC_PAD > 0
    ulong pad[ATOMIC_PAD];
#endif
} rtcsample_t;

// The management structure
// All bits 0 is an acceptable state, and the expected initial state
typedef struct heap_s {
    start_t start[NUM_KINDS];                      // Used to guide the search for a slab to allocate from
    nallocated_t num_allocated_slabs[NUM_KINDS];   // The number of allocated slabs of a given kind
    nrecordable_t num_recordable_slabs[NUM_KINDS]; // The number of slabs that can be recorded (a multiple of NUM_SDATA)
    rtcsample_t salloc_time[NUM_KINDS];            // The time the most recent slab allocation was started
    rtcsample_t grow_time[NUM_KINDS];              // The time the most recent grow recordable was started
    sdata_t sdata[NUM_KINDS][NUM_SDATA];           // Information about all allocated slabs
    atomic_ulong initial_slabs;                    // Next initial slab to deliver
    ulong initial_slabs_end;                       // End of inititial slabs
    ulong initial_slabs_start;                     // Start of initial slabs
#if defined NON_SLAB_TRACKING
#if ATOMIC_PAD > 1
    ulong pad[ATOMIC_PAD-1];
#endif
    atomic_ulong num_nonslab_allocations;          // Count of number of non-slab allocations that have not been freed
#endif
} heap_t;

// Atomics wrappers
#define AL(P, O) __opencl_atomic_load(P, O, memory_scope_device)
#define AS(P, V, O) __opencl_atomic_store(P, V, O, memory_scope_device)
#define AFA(P, V, O) __opencl_atomic_fetch_add(P, V, O, memory_scope_device)
#define AFS(P, V, O) __opencl_atomic_fetch_sub(P, V, O, memory_scope_device)
#define AFN(P, V, O) __opencl_atomic_fetch_and(P, V, O, memory_scope_device)
#define AFO(P, V, O) __opencl_atomic_fetch_or (P, V, O, memory_scope_device)
#define ACE(P, E, V, O) __opencl_atomic_compare_exchange_strong(P, E, V, O, O, memory_scope_device)

// get the heap pointer
static __global heap_t *
get_heap_ptr(void) {
    if (__oclc_ABI_version < 500) {
        static __global heap_t heap;
        return &heap;
    } else {
        return (__global heap_t *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[12];
    }
}

// The actual number of blocks in a slab with blocks of kind k
static uint
num_blocks(kind_t k)
{
    return kinfo[k].num_blocks;
}

// The usable number of blocks in a slab with blocks of kind k
static uint
num_usable_blocks(kind_t k)
{
    return kinfo[k].num_usable_blocks;
}

// The number of used blocks in a slab of kind k triggering skipping while searching
static uint
skip_threshold(kind_t k)
{
    return kinfo[k].skip_threshold;
}

// The offset to the first block in a slab of kind k
static uint
block_offset(kind_t k)
{
    return kinfo[k].block_offset;
}

// The index of the first unusable block in a slab of kind k
static uint
first_unusable(kind_t k)
{
    return kinfo[k].first_unusable;
}

// The gap or distance between indices of unusable blocks in a slab of kind k
static uint
gap_unusable(kind_t k)
{
    return kinfo[k].gap_unusable;
}

// The pattern of unusable bits when the gap is less than 32
static uint
pattern_unusable(kind_t k)
{
    return kinfo[k].pattern_unusable;
}

// The number of active lanes at this point
static uint
active_lane_count(void)
{
    if (__oclc_wavefrontsize64) {
        return __builtin_popcountl(__builtin_amdgcn_read_exec());
    } else {
        return __builtin_popcount(__builtin_amdgcn_read_exec_lo());
    }
}

// Overloads to broadcast the value held by the first active lane
// The result is known to be wave-uniform
static __attribute__((overloadable)) uint
first(uint v)
{
    return __builtin_amdgcn_readfirstlane(v);
}

static __attribute__((overloadable)) ulong
first(ulong v)
{
    uint2 v2 = __builtin_astype(v, uint2);
    uint2 w2;
    w2.x = __builtin_amdgcn_readfirstlane(v2.x);
    w2.y = __builtin_amdgcn_readfirstlane(v2.y);
    return __builtin_astype(w2, ulong);
}

static __attribute__((overloadable)) __global void *
first(__global void * v)
{
    uint2 v2 = __builtin_astype(v, uint2);
    uint2 w2;
    w2.x = __builtin_amdgcn_readfirstlane(v2.x);
    w2.y = __builtin_amdgcn_readfirstlane(v2.y);
    return __builtin_astype(w2, __global void *);
}

// Count the number of true arguments across the wave
static uint
votes(bool b)
{
    return __builtin_popcountl(__builtin_amdgcn_ballot_w64(b));
}

// Return mask of lanes with identical arguments
static __attribute__((overloadable)) ulong
match(__global void *p)
{
    ulong ret = 0;
    int go = 1;
    do {
        if (go) {
            __global void *fp = first(p);
            if (p == fp) {
                ret = __builtin_amdgcn_ballot_w64(1);
                go = 0;
            }
        }
    } while (__builtin_amdgcn_ballot_w64(go) != 0);

    return ret;
}

static __attribute__((overloadable)) ulong
match(kind_t k)
{
    ulong ret = 0;
    int go = 1;
    do {
        if (go) {
            kind_t fk = first(k);
            if (k == fk) {
                ret = __builtin_amdgcn_ballot_w64(1);
                go = 0;
            }
        }
    } while (__builtin_amdgcn_ballot_w64(go) != 0);

    return ret;
}

// Broadcast the value in a lane to the group
static __attribute__((overloadable)) __global void *
gcast(__global void *p, uint l)
{
    uint2 p2 = AS_UINT2(p);
    uint2 r;
    r.x = __builtin_amdgcn_ds_bpermute(l << 2, p2.x);
    r.y = __builtin_amdgcn_ds_bpermute(l << 2, p2.y);
    return (__global void *)AS_ULONG(r);
}

static __attribute__((overloadable)) ulong
gcast(ulong v, uint l)
{
    uint2 v2 = AS_UINT2(v);
    uint2 r;
    r.x = __builtin_amdgcn_ds_bpermute(l << 2, v2.x);
    r.y = __builtin_amdgcn_ds_bpermute(l << 2, v2.y);
    return AS_ULONG(r);
}

static __attribute__((overloadable)) uint
gcast(uint v, uint l)
{
    return __builtin_amdgcn_ds_bpermute(l << 2, v);
}

static uint
leader(gmask_t gm)
{
    return (uint)BUILTIN_CTZ_U64(gm);
}

static uint
position(gmask_t gm)
{
  return __builtin_amdgcn_mbcnt_hi(gm >> 32, __builtin_amdgcn_mbcnt_lo((uint)gm, 0u));
}

static uint
members(gmask_t gm)
{
    return __builtin_popcountl(gm);
}

static gmask_t
before(gmask_t gm, uint l)
{
    return gm & ((1UL << l) - 1UL);
}

// The kind of the smallest block that can hold sz bytes
static uint
size_to_kind(uint sz)
{
    sz = sz < 16 ? 16 : sz;
    uint b = 31 - OCKL_MANGLE_U32(clz)(sz);
    uint v = 1 << b;
    return ((b - 4) << 1) + (sz > v) + (sz > (v | (v >> 1)));
}

// The size of a block of kind k
// Alternatively we could place this in kinfo
static uint
kind_to_size(kind_t k)
{
    uint s = 1 << ((k >> 1) + 4);
    return s + ((k & 1) != 0 ? (s >> 1) : 0);
}

// Get the sdata pointer corresponding to kind k and index i
// Assumes only 2 levels
static __global sdata_t *
sdata_for(__global heap_t *hp, kind_t k, sid_t i)
{
    if (i >= NUM_SDATA) {
        i -= NUM_SDATA;
        __global sdata_t *sdp = &hp->sdata[k][i >> SDATA_SHIFT];
        ulong array = AL(&sdp->array, memory_order_relaxed);
        __global sdata_t *sda = (__global sdata_t *)array;
        return &sda[i & SDATA_MASK];
    } else {
        return &hp->sdata[k][i];
    }
}

// Get the sdata parent pointer corresponding to kind k and index i
// Also assumes only 2 levels, and i must be >= NUM_SDATA
static __global sdata_t *
sdata_parent_for(__global heap_t *hp, kind_t k, sid_t i)
{
    return &hp->sdata[k][(i - NUM_SDATA) >> SDATA_SHIFT];
}

// Free a non-slab allocation
static void
non_slab_free(ulong addr)
{
    __ockl_devmem_request(addr, 0);

#if defined NON_SLAB_TRACKING
    uint aid = __ockl_activelane_u32();
    uint nactive = active_lane_count();

    if (aid == 0) {
        __global heap_t *hp = get_heap_ptr();
        AFS(&hp->num_nonslab_allocations, nactive, memory_order_relaxed);
    }
#endif
}

// public dealloc() entrypoint
__attribute__((cold)) void
__ockl_dm_dealloc(ulong addr)
{
    if ((addr & 0xfffUL) == 0UL) {
        if (addr)
            non_slab_free(addr);
        return;
    }

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent", "global");
 
    __global heap_t *hp = get_heap_ptr();

    ulong saddr = addr & ~(ulong)0x1fffffUL;
    __global slab_t *sptr = (__global slab_t *)saddr;

    kind_t k = sptr->k;
    sid_t i = sptr->i;

    gmask_t gm = match(sptr);
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    __global sdata_t *sdp = 0;
    if (l == ll)
        sdp = sdata_for(hp, k, i);
    sdp = gcast(sdp, ll);

    uint b = (uint)(addr - (saddr + block_offset(k))) / kind_to_size(k);
    AFN(&sptr->in_use[b >> 5], ~(1 << (b & 0x1f)), memory_order_relaxed);
    if (l == ll)
        AFS(&sdp->num_used_blocks, members(gm), memory_order_relaxed);
}

// The is the malloc implementation for sizes greater
// than ALLOC_THRESHOLD
static __global void *
non_slab_malloc(size_t sz)
{
    ulong addr = __ockl_devmem_request(0, sz);

#if defined NON_SLAB_TRACKING
    if (addr != 0) {
        uint aid = __ockl_activelane_u32();
        uint nactive = active_lane_count();

        if (aid == 0) {
            __global heap_t *hp = get_heap_ptr();
            AFA(&hp->num_nonslab_allocations, nactive, memory_order_relaxed);
        }
    }
#endif

    return (__global void *)addr;
}

// Wait for a while to let a new slab of kind k to appear
static void
new_slab_wait(__global heap_t *hp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);
    if (l == ll) {
        ulong expected = AL(&hp->salloc_time[k].value, memory_order_relaxed);
        ulong now = __ockl_steadyctr_u64();
        ulong dt = now - expected;
        if  (dt < SLAB_TICKS)
            __ockl_rtcwait_u32(SLAB_TICKS - (uint)dt);
    }
}

// Wait for a while to let the number of recordable slabs of kind k to grow
static void
grow_recordable_wait(__global heap_t *hp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    if (l == ll) {
        ulong expected = AL(&hp->grow_time[k].value, memory_order_relaxed);
        ulong now = __ockl_steadyctr_u64();
        ulong dt = now - expected;
        if  (dt < GROW_TICKS)
            __ockl_rtcwait_u32(GROW_TICKS - (uint)dt);
    }
}

// Wait to let a CAS failure clear
static void
cas_wait(void)
{
    __builtin_amdgcn_s_sleep(CAS_SLEEP);
}

// Obtain a new sdata array
// Expect only one active lane here
static ulong
obtain_new_array(void)
{
    return __ockl_devmem_request(0, sizeof(sdata_t) * NUM_SDATA);
}

// Clear an array of sdata
static void
clear_array(ulong a, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);
    uint n = members(gm);
    __global ulong *p = (__global ulong *)a;

    for (uint i = position(gm); i < NUM_SDATA*ULONG_PER_SDATA; i += n)
        p[i] = 0UL;
}

// Release an array
// Expect only one active lane here
static void
release_array(ulong a)
{
    __ockl_devmem_request(a, 0);
}

// Try to grow the number of recordable slabs
// The arguments and result are uniform
static uint
try_grow_num_recordable_slabs(__global heap_t *hp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    uint nrs = 0;
    if (l == ll)
        nrs = AL(&hp->num_recordable_slabs[k].value, memory_order_relaxed);
    nrs = gcast(nrs, ll);

    if (nrs == MAX_RECORDABLE_SLABS)
        return GROW_FAILURE;

    uint ret = GROW_BUSY;
    if (l == ll) {
        ulong expected = AL(&hp->grow_time[k].value, memory_order_relaxed);
        ulong now = __ockl_steadyctr_u64();
        if (now - expected >= GROW_TICKS &&
            ACE(&hp->grow_time[k].value, &expected, now, memory_order_relaxed))
                ret = GROW_FAILURE;
    }
    ret = gcast(ret, ll);

    if (ret == GROW_BUSY)
        return ret;

    ulong sa = 0;
    if (l == ll)
        sa = obtain_new_array();
    sa = gcast(sa, ll);

    if (!sa)
        return ret;

    clear_array(sa, gm);


    for (;;) {
        if (l == ll)
            nrs = AL(&hp->num_recordable_slabs[k].value, memory_order_relaxed);
        nrs = gcast(nrs, ll);

        if (nrs == MAX_RECORDABLE_SLABS) {
            if (l == ll)
                release_array(sa);
            return ret;
        }

        if (l == ll) {
            __global sdata_t *sdp = sdata_parent_for(hp, k, nrs);

            ulong expected = 0UL;
            bool done = ACE(&sdp->array, &expected, sa, memory_order_relaxed);
            ret = done ? GROW_SUCCESS : ret;
            if (done)
                AFA(&hp->num_recordable_slabs[k].value, NUM_SDATA, memory_order_release);
        }
        ret = gcast(ret, ll);

        if (ret == GROW_SUCCESS)
            return ret;

        cas_wait();
    }
}

// Obtain a new slab
// Only expect one lane active here
static ulong
obtain_new_slab(__global heap_t *hp)
{
    ulong is = AL(&hp->initial_slabs, memory_order_relaxed);
    ulong se = hp->initial_slabs_end;
    if (is < se) {
        is = AFA(&hp->initial_slabs, 1UL << 21, memory_order_relaxed);
        if (is < se)
            return is;
    }
    ulong ret = __ockl_devmem_request(0, 1UL << 21);
    return ret;
}

// Initialize a slab
// Rely on the caller to release the changes
static void
initialize_slab(__global slab_t *s, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);
    uint j = position(gm);
    uint di = members(gm);

    uint g = gap_unusable(k);
    uint m = num_blocks(k);
    uint n = (m + 31) >> 5;

    __global uint *p = (__global uint *)&s->in_use;
    if (g > 32) {
        for (uint i = j; i < n; i += di)
            p[i] = 0;

        di *= g;
        for (uint i = first_unusable(k) + j*g; i < m; i += di)
            p[i >> 5] = 1 << (i & 0x1f);
    } else {
        uint v = pattern_unusable(k);
        for (uint i = j; i < n; i += di)
            p[i] = v;
    }

    if (l == ll) {
        uint mm = m & 0x1f;
        if (mm != 0)
            p[n-1] |= ~0 << mm;

        *((__global uint4 *)s) = (uint4)(k, 0, 0, 0);
    }
}

// Release a slab
// Only expect one lane active here
static void
release_slab(ulong saddr)
{
    __ockl_devmem_request(saddr, 0);
}

// Try to allocate a new slab of kind k
static __global sdata_t *
try_allocate_new_slab(__global heap_t *hp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    for (;;) {
        uint nas = 0;
        if (l == ll)
            nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
        nas = gcast(nas, ll);

        if (nas == MAX_RECORDABLE_SLABS)
            return (__global sdata_t *)0;

        uint nrs = 0;
        if (l == ll)
            nrs = AL(&hp->num_recordable_slabs[k].value, memory_order_relaxed);
        nrs = gcast(nrs, ll);

        if (nas == nrs) {
            uint result = try_grow_num_recordable_slabs(hp, k, gm);
            if (result != GROW_SUCCESS) {
                grow_recordable_wait(hp, k, gm);
                return result == GROW_FAILURE ? (__global sdata_t *)0 : SDATA_BUSY;
            }
        }

        __global sdata_t *ret = SDATA_BUSY;
        if (l == ll) {
            ulong expected = AL(&hp->salloc_time[k].value, memory_order_relaxed);
            ulong now = __ockl_steadyctr_u64();
            if (now - expected >= SLAB_TICKS &&
                ACE(&hp->salloc_time[k].value, &expected, now, memory_order_relaxed))
                ret = (__global sdata_t *)0;
        }
        ret = gcast(ret, ll);

        if (ret)
            return ret;

        ulong saddr = 0;
        if (l == ll)
            saddr = obtain_new_slab(hp);
        saddr = gcast(saddr, ll);

        if (!saddr)
            return (__global sdata_t *)0;

        initialize_slab((__global slab_t *)saddr, k, gm);

        for (;;) {
            if (l == ll)
                nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
            nas = gcast(nas, ll);

            if (nas == MAX_RECORDABLE_SLABS)
                return (__global sdata_t *)0;

            if (l == ll)
                nrs = AL(&hp->num_recordable_slabs[k].value, memory_order_relaxed);
            nrs = gcast(nrs, ll);

            if (nas == nrs) {
                if (l == ll)
                    release_slab(saddr);
                break;
            }

            if (l == ll) {
                ret = sdata_for(hp, k, nas);
                AS(&ret->num_used_blocks, members(gm), memory_order_relaxed);
                ((__global slab_t *)saddr)->i = nas;
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent", "global");
                ulong expected = 0;
                bool done = ACE(&ret->saddr, &expected, saddr, memory_order_relaxed);
                ret = done ? ret : (__global sdata_t *)0;
                if (done)
                    AFA(&hp->num_allocated_slabs[k].value, 1, memory_order_release);
            }
            ret = gcast(ret, ll);

            if (ret)
                return ret;

            cas_wait();
        }
    }
}

static __global sdata_t *
normal_slab_find(__global heap_t *hp, kind_t k, gmask_t gm, uint nas)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);
    uint n = members(gm);

    for (;;) {
        if (nas > 0) {
            int nleft = (int)nas;

            uint start = 0;
            if (l == ll)
                start = AL(&hp->start[k].value, memory_order_relaxed);
            start = gcast(start, ll);
            uint i = start;

            do {
                __global sdata_t *sdp = sdata_for(hp, k, (i + position(gm)) % nas);
                uint nub = AL(&sdp->num_used_blocks, memory_order_relaxed);
                ulong sm = __builtin_amdgcn_ballot_w64(nub < skip_threshold(k)) & gm;
                if (sm) {
                    if (l == ll) {
                        uint j = (i +  members(before(gm, leader(sm)))) % nas;
                        sdp =  sdata_for(hp, k, j);
                        nub = AFA(&sdp->num_used_blocks, n, memory_order_relaxed);
                        if (nub + n > num_usable_blocks(k)) {
                            AFS(&sdp->num_used_blocks, n, memory_order_relaxed);
                            sdp = (__global sdata_t *)0;
                        }
                        if (sdp && j != start)
                             AS(&hp->start[k].value, j, memory_order_relaxed);
                    }
                    sdp = gcast(sdp, ll);
                    if (sdp)
                        return sdp;
                    cas_wait();
                } else {
                    i += n;
                    nleft -= (int)n;
                }
            } while (nleft > 0);
        }

        __global sdata_t *sdp = try_allocate_new_slab(hp, k, gm);
        if (sdp != SDATA_BUSY)
            return sdp;

        new_slab_wait(hp, k, gm);
        if (l == ll)
            nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
        nas = gcast(nas,ll);
    }
}

// Find a slab of kind k that can be searched for blocks using
// the "final" approach.  The arguments and results are uniform
static __global sdata_t *
final_slab_find(__global heap_t *hp, kind_t k0, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);
    int n = (int)members(gm);

    for (kind_t k = k0;;) {
        int nleft = MAX_RECORDABLE_SLABS;
        uint i = 0;
        if (l == ll)
            i = AL(&hp->start[k].value, memory_order_relaxed);
        i = gcast(i, ll);
        do {
            __global sdata_t *sdp = sdata_for(hp, k, (i + position(gm)) % MAX_RECORDABLE_SLABS);
            uint nub = AL(&sdp->num_used_blocks, memory_order_relaxed);
            ulong sm = __builtin_amdgcn_ballot_w64(nub < num_usable_blocks(k) - n) & gm;
            if (sm) {
                if (l == ll) {
                    uint j = (i + leader(sm)) % MAX_RECORDABLE_SLABS;
                    sdp =  sdata_for(hp, k, j);
                    nub = AFA(&sdp->num_used_blocks, n, memory_order_relaxed);
                    if (nub > num_usable_blocks(k)) {
                        AFS(&sdp->num_used_blocks, n, memory_order_relaxed);
                        sdp = (__global sdata_t *)0;
                    }
                    if (sdp)
                         AS(&hp->start[k].value, j, memory_order_relaxed);
                }
                sdp = gcast(sdp, ll);
                if (sdp)
                    return sdp;
                cas_wait();
            } else {
                i += n;
                nleft -= n;
            }
        } while (nleft > 0);

        uint nextk = k + 2 - (k & 1);

        if (k != k0 || nextk >= NUM_KINDS)
            return (__global sdata_t *)0;

        uint nas = 0;
        if (l == ll)
            nas = AL(&hp->num_allocated_slabs[nextk].value, memory_order_relaxed);
        nas = gcast(nas, ll);

        if (nas < MAX_RECORDABLE_SLABS)
            return normal_slab_find(hp, nextk, gm, nas);

        k = nextk;
    }
}

// Find a slab of kind k that can be searched for blocks
// The arguments and results are uniform
static __global sdata_t *
slab_find(__global heap_t *hp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    uint nas = 0;
    if (l == ll)
        nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
    nas = gcast(nas, ll);

    if (nas < MAX_RECORDABLE_SLABS)
        return normal_slab_find(hp, k, gm, nas);
    else
        return final_slab_find(hp, k, gm);
}

static __global void *
block_find(__global sdata_t *sdp, kind_t k, gmask_t gm)
{
    uint l = __ockl_lane_u32();
    uint ll = leader(gm);

    __global slab_t *sp = 0;
    uint i = 0;
    if (l == ll) {
        sp = (__global slab_t *)AL(&sdp->saddr, memory_order_relaxed);
        i = AFA(&sp->start, members(gm), memory_order_relaxed);
    }
    sp = gcast(sp, ll);
    i = gcast(i, ll);
    i = (((i + position(gm)) << 5) % num_blocks(k)) >> 5;

    uint n = (num_blocks(k) + 31) >> 5;
    for (;;) {
        __global atomic_uint *p = sp->in_use + i;
        uint m = AL(p, memory_order_relaxed);
        if (m != ~0) {
            uint b = BUILTIN_CTZ_U32(~m);
            uint mm = AFO(p, 1 << b, memory_order_relaxed);
            if ((mm & (1 << b)) == 0) {
                uint ii = (i << 5) + b;
                return (__global void *)((__global char *)sp + block_offset(k) + kind_to_size(k)*ii);
            }
        }
        i = (i + 1) % n;
    }
}

// This is the malloc implementation for sizes that fit in some kind of block
static __global void *
slab_malloc(int sz)
{
    kind_t k = size_to_kind(sz);
    __global heap_t *hp = get_heap_ptr();
    gmask_t gm = match(k);

    __global sdata_t *sdp = slab_find(hp, k, gm);
    if (sdp != (__global sdata_t *)0)
        return block_find(sdp, k, gm);

    return (__global void *)0;
}

// public alloc() entrypoint
__attribute__((cold)) __global void *
__ockl_dm_alloc(ulong sz)
{
    if (sz == 0)
        return (__global void *)0;

    if (sz > ALLOC_THRESHOLD)
        return non_slab_malloc(sz);

    return slab_malloc(sz);
}

// Initialize the heap
//   This is intended to be called by a kernel launched by the language runtime
//   at device initialization time. The launched NDrange must have one workgroup
//   consisting of 256 workitems.
__attribute__((weak)) void
__ockl_dm_init_v1(ulong hp, ulong sp, uint hb, uint nis)
{
    uint lid = __ockl_get_local_id(0);

    // 0 is used to indicate no clearing needed
    if (hb) {
        __global int4 *p = (__global int4 *)(hp + lid*16);
        for (int i=0; i<131072/16/256; ++i) {
            *p = (int4)0;
            p += 256;
        }
    }

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent", "global");
    __builtin_amdgcn_s_barrier();

    __global heap_t *thp = (__global heap_t *)hp;

    if (lid < NUM_KINDS)
        AS(&thp->num_recordable_slabs[lid].value, NUM_SDATA, memory_order_relaxed);

    if (lid == 0) {
        AS(&thp->initial_slabs, sp, memory_order_relaxed);
        thp->initial_slabs_end = sp + ((ulong)nis << 21);
        thp->initial_slabs_start = sp;
    }
}

// reverse local array, n <= wavesize
// Expect this to be called by one full wave
// TODO make this work on devices which can't permute full wave
static void __attribute__((target("gfx8-insts")))
reverse_la(__local uint *x, uint i, uint n)
{
    if (i < n) {
        uint j = n - 1 - i;
        x[i] = __builtin_amdgcn_ds_bpermute(j << 2, x[i]);
    }
}

// Shift wavesize consecutive elements downward by n
static void
shift_la(__local uint *a, uint i, uint n)
{
    a[i] = a[i+n];
}

// Find and record destination location for trim
static uint
dst_scan(__global heap_t *hp, kind_t k, ulong iss, ulong ise, uint l, uint i, uint n, uint c0, __local uint *d)
{
    bool b = false;

    if (l+i < n) {
        __global sdata_t *sdp = sdata_for(hp, k, l+i);
        uint nub = AL(&sdp->num_used_blocks, memory_order_relaxed);
        ulong saddr = AL(&sdp->saddr, memory_order_relaxed);

        b = nub == 0 && saddr && (saddr < iss || saddr >= ise);
        if (b) {
            release_slab(saddr);
            AS(&sdp->saddr, 0UL, memory_order_relaxed);
            AS(&sdp->num_used_blocks, 0U, memory_order_relaxed);
            d[c0+__ockl_activelane_u32()] = l+i;
        }
    }

    return c0 + votes(b);
}

// Find and record source location for trim
static uint
src_scan(__global heap_t *hp, kind_t k, ulong iss, ulong ise, uint r, uint i, uint n, uint c0, __local uint *s)
{
    bool b = false;

    if (r+i < n) {
        __global sdata_t *sdp = sdata_for(hp, k, r+i);
        ulong saddr = AL(&sdp->saddr, memory_order_relaxed);
        uint nub = AL(&sdp->num_used_blocks, memory_order_relaxed);

        b = nub > 0 || (saddr >= iss && saddr < ise);
        if (b) {
            s[c0+__ockl_activelane_u32()] = r+i;
        } else if (saddr) {
            release_slab(saddr);
            AS(&sdp->saddr, 0UL, memory_order_relaxed);
            AS(&sdp->num_used_blocks, 0U, memory_order_relaxed);
        }
    }

    uint c = votes(b);
    reverse_la(s + c0, i, c);
    return c0 + c;
}

// Count available slabs
static uint
end_scan(__global heap_t *hp, kind_t k, uint l, uint i, uint n, int c0)
{
    bool b = false;

    if (l+i < n) {
        __global sdata_t *sdp = sdata_for(hp, k, l+i);
        ulong saddr = AL(&sdp->saddr, memory_order_relaxed);
        b = saddr != 0;
    }

    return c0 + votes(b);
}

// Move up to n slabs (n <= wavesize) from index in s[] to index in d[]
// and return the number moved
static uint
move_slabs(__global heap_t *hp, kind_t k, uint i, uint n, __local uint *d, __local uint *s)
{
    bool b = i < n && d[i] < s[i];
    if (b) {
        __global sdata_t *dsdp = sdata_for(hp, k, d[i]);

        __global sdata_t *ssdp = sdata_for(hp, k, s[i]);
        ulong ssaddr = AL(&ssdp->saddr, memory_order_relaxed);
        ((__global slab_t *)ssaddr)->i = d[i];

        AS(&dsdp->saddr, ssaddr, memory_order_relaxed);
        AS(&dsdp->num_used_blocks, AL(&ssdp->num_used_blocks, memory_order_relaxed), memory_order_relaxed);

        AS(&ssdp->saddr, 0UL, memory_order_relaxed);
        AS(&ssdp->num_used_blocks, 0UL, memory_order_relaxed);
    }

    return votes(b);
}

// "Trim" slabs of kind k
// Expecting an exactly one-full-wave caller
static uint
trim_kind(__global heap_t *hp, kind_t k, ulong iss, ulong ise, uint i, uint n, __local uint *srcs, __local uint *dsts)
{
    uint l = 0;
    uint lm = 0;
    uint nd = 0;
    const uint wsz = __oclc_wavefrontsize64 ? 64 : 32;

    uint r = (n - 1) / wsz * wsz;
    uint ns = 0;

    for (;;) {
        while (l < n && nd < wsz) {
            nd = dst_scan(hp, k, iss, ise, l, i, n, nd, dsts);
            l += wsz;
        }

        if (nd == 0)
            break;

        while (r < n && ns < wsz) {
            ns = src_scan(hp, k, iss, ise, r, i, n, ns, srcs);
            r -= wsz;
        }

        if (ns == 0)
            break;

        uint m = nd < ns ? nd : ns;
        m = wsz < m ? wsz : m;

        uint mm = move_slabs(hp, k, i, m, dsts, srcs);

        if (mm)
            lm = dsts[mm-1];

        if (l >= n || mm != m)
            break;

        shift_la(dsts, i, m);
        shift_la(srcs, i, m);
        nd -= m;
        ns -= m;
    }

    lm = lm / wsz * wsz;
    l = lm;
    uint nn = lm;
    do {
        nn = end_scan(hp, k, l, i, n, nn);
        l += wsz;
    } while (l == nn);

    return nn;
}

// "Trim" non-initial empty slabs of all kinds
//
// This function must be called from a 1D 1-full-wave kernel that only 
// calls this function.  When that kernel runs, no other kernel on the
// device using dm_[de]alloc may be running.
//
// The calling kernel must pass in a generic pointer to a __local int array with 4*wavesize elements
//
// TODO consider a design which allows trimming concurrent with other use
//
__attribute__((weak, cold)) void
__ockl_dm_trim(int *mem)
{
    __local uint *dsts = (__local uint *)mem;
    __local uint *srcs = dsts + (__oclc_wavefrontsize64 ? 2*64 : 2*32);
    __global heap_t *hp = get_heap_ptr();
    ulong iss = hp->initial_slabs_start;
    ulong ise = hp->initial_slabs_end;
    uint i = __ockl_lane_u32();

    for (kind_t k=0; k<NUM_KINDS; ++k) {
        uint nas = 0;
        if (i == 0)
            nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
        nas = first(nas);
        if (nas) {
            uint tnas = trim_kind(hp, k, iss, ise, i, nas, srcs, dsts);
            if (i == 0)
                AS(&hp->num_allocated_slabs[k].value, tnas, memory_order_relaxed);
        }
    }
}

// Grab some info about the current state of the heap
// Expecting the caller to limit the number of threads executing here to 1
__attribute__((cold)) void
__ockl_dm_hinfo(ulong *rp)
{
    __global heap_t *hp = get_heap_ptr();

    *rp++ = NUM_KINDS;
    for (kind_t k=0; k<NUM_KINDS; ++k) {
        uint nas = AL(&hp->num_allocated_slabs[k].value, memory_order_relaxed);
        *rp++ = (ulong)nas;
        ulong nubs = 0;
        for (uint i = 0; i<nas; ++i) {
            __global sdata_t *sdp = sdata_for(hp, k, i);
            uint nub = AL(&sdp->num_used_blocks, memory_order_relaxed);
            nubs += nub;
        }
        *rp++ = nubs;
        *rp++ = (ulong)nas * num_usable_blocks(k);
    }
#if defined NON_SLAB_TRACKING
    *rp++ = AL(&hp->num_nonslab_allocations, memory_order_relaxed);
#else
    *rp++ = 0;
#endif
}

// 

#if defined NON_SLAB_TRACKING
// return a snapshot of the current number of nonslab allocations
// which haven't been deallocated
__attribute__((cold)) ulong
__ockl_dm_nna(void)
{
    __global heap_t *hp = get_heap_ptr();
    return AL(&hp->num_nonslab_allocations, memory_order_relaxed);
}
#endif

