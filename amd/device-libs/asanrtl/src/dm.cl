/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "asan_util.h"
#include "shadow_mapping.h"

static const __constant uint kAsanHeapLeftRedzoneMagicx4 = 0xfafafafaU;
static const __constant ulong kAsanHeapLeftRedzoneMagicx8 = 0xfafafafafafafafaUL;
static const __constant uchar kAsanHeapFreeMagic = (uchar)0xfd;

extern ulong __ockl_devmem_request(ulong addr, ulong size);

// Minimum Number of bytes we want to quarantine
#define QUARANTINE_BYTES (SLAB_BYTES * 16)

// Whether we track non-slab allocations
#define NON_SLAB_TRACKING 1

// Magic at beginning of allocation
#define ALLOC_MAGIC 0xfedcba1ee1abcdefUL

#define AS(P,V) __opencl_atomic_store(P, V, memory_order_relaxed, memory_scope_device)
#define AL(P) __opencl_atomic_load(P, memory_order_relaxed, memory_scope_device)
#define AA(P,V) __opencl_atomic_fetch_add(P, V, memory_order_relaxed, memory_scope_device)
#define AO(P,V) __opencl_atomic_fetch_or(P, V, memory_order_relaxed, memory_scope_device)
#define ACE(P,E,V) __opencl_atomic_compare_exchange_strong(P, E, V, memory_order_relaxed, memory_order_relaxed, memory_scope_device)

// An allocation
#define ALLOC_HEADER_BYTES 32
typedef struct alloc_struct {
    ulong magic;   // Assist with memory scan for header
    ulong sp;      // slab pointer, 0 if non-slab allocation
    ulong pc;      // We can only collect PC currently, callstack ID later
    uint asz;      // Total number of bytes including header, redzone, and round, multiple of 16
    uint usz;      // user specificed size
    ulong ret[];   // Address returned by malloc, always 16-byte aligned
} alloc_t;

// Assumes 4096 byte minimum alignment of slab
#define SLAB_ALIGN 4096
#define SLAB_CTR_MASK (ulong)(SLAB_ALIGN - 1)
#define SLAB_BUSY ((__global slab_t *)1UL)
#define SLAB_TICKS 20000
#define SLAB_BYTES (1UL << 21)
#define SLAB_THRESHOLD (SLAB_BYTES / 64)
#define SLAB_HEADER_BYTES 32
#define SLAB_RECYCLE_THRESHOLD ((QUARANTINE_BYTES+SLAB_BYTES-1) / SLAB_BYTES)

// A slab of memory used to provide malloc returned blocks
typedef struct slab_s {
    atomic_ulong next;   // link to next slab on queue chain, must be first
    atomic_ulong next2;  // link to next slab on stack chain, must be second
    atomic_ulong ap;     // Pointer to next allocation (>= &space[0] )
    atomic_uint rb;      // returned bytes
    atomic_uint flags;   // flags
    ulong space[(SLAB_BYTES-SLAB_HEADER_BYTES)/8];  // Space for allocations.  Must  be aligned 16
} slab_t;

// The heap
typedef struct heap_s {
    atomic_ulong fake_next;               // Heap is a fake slab, must be first
    atomic_ulong fake_next2;              // Heap is a fake slab, must be second
    atomic_ulong head;                    // points to dummy or most recently dequeued slab
    atomic_ulong tail;                    // usually points to most recently enqueued slab
    atomic_ulong top;                     // Top of slab stack
    atomic_ulong cs;                      // current slab pointer
    atomic_ulong atime;                   // Time most recent allocation started
    atomic_ulong initial_slabs;           // pointer to next preallocated slab
    ulong initial_slabs_end;              // pointer to end of preallocated slabs
    atomic_uint nas;                      // Number of allocated slabs
#if defined NON_SLAB_TRACKING
    atomic_ulong num_nonslab_allocations; // Count of number of non-slab allocations that have not been freed
#endif
} heap_t;

// Inhibit control flow optimizations
#define O0(X) X = o0(X)
__attribute__((overloadable)) static int o0(int x) { int y; __asm__ volatile("" : "=v"(y) : "0"(x)); return y; }
__attribute__((overloadable)) static uint o0(uint x) { uint y; __asm__ volatile("" : "=v"(y) : "0"(x)); return y; }
__attribute__((overloadable)) static ulong o0(ulong x) { ulong y; __asm__ volatile("" : "=v"(y) : "0"(x)); return y; }

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

// The number of active lanes at this point
static uint
active_lane_count_w64(void)
{
    return __builtin_popcountl(__builtin_amdgcn_read_exec());
}

static uint
active_lane_count_w32(void)
{
    return __builtin_popcount(__builtin_amdgcn_read_exec_lo());
}

static uint
active_lane_count(void)
{
    return __oclc_wavefrontsize64 ? active_lane_count_w64() : active_lane_count_w32();
}

static ulong
round_16(ulong n)
{
    return ((n + 15) >> 4) << 4;
}

static __global slab_t *
slabptr(ulong p)
{
    return (__global slab_t *)(p & ~SLAB_CTR_MASK);
}

static ulong
addcnt(ulong p, ulong c)
{
    return p | (((c & SLAB_CTR_MASK) + 1UL) & SLAB_CTR_MASK);
}

NO_SANITIZE_ADDR
static __global heap_t *
get_heap_ptr(void) {
    if (__oclc_ABI_version < 500) {
        static __attribute__((aligned(4096))) __global heap_t heap;
        return &heap;
    } else {
        return (__global heap_t *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[12];
    }
}

// Size of additional left redzone, roughly assumes 32 byte header, multiple of 16
static uint
added_redzone(uint sz)
{
    return sz < 128 ? 0 :
           sz < 512 ? 96 :
           sz < 2048 ? 224 :
           sz < 8192 ? 992 : 2016;
}

// Called by a single workitem
static void
slab_pause(void)
{
    __builtin_amdgcn_s_sleep(2);
}

// Intended to be called from only one lane of a wave
NO_SANITIZE_ADDR
static void
put_free_slab(__global heap_t *hp, __global slab_t *sp)
{
    ulong head = AL(&hp->head);
    if (slabptr(head) == sp) {
        ulong top = AL(&hp->top);
        for (;;) {
          AS(&sp->next2, (ulong)slabptr(top));
          if (ACE(&hp->top, &top, addcnt((ulong)sp, top)))
              return;
          slab_pause();
        }
    }
    AS(&sp->next, 0UL);

    ulong tail = AL(&hp->tail);
    for (;;) {
        __global slab_t *last = slabptr(tail);
        ulong next = 0;
        if (ACE(&last->next, &next, (ulong)sp))
            break;

        ACE(&hp->tail, &tail, addcnt(next, tail));
        slab_pause();
    }

    ACE(&hp->tail, &tail, addcnt((ulong)sp, tail));
    return;
}

// Intended to be called from only one lane of a wave
NO_SANITIZE_ADDR
static __global slab_t *
get_free_slab(__global heap_t *hp)
{
    for (;;) {
        ulong head = AL(&hp->head);
        __global slab_t *first = slabptr(head);
        ulong next = AL(&first->next);
        if (head == AL(&hp->head)) {
            ulong tail = AL(&hp->tail);
            if (first == slabptr(tail)) {
                if (!next)
                    break;
                ACE(&hp->tail, &next, addcnt(next, tail));
            } else if (next) {
                if (ACE(&hp->head, &head, addcnt(next, head)))
                    return slabptr(next);
            }
        }
        slab_pause();
    }

    ulong top = AL(&hp->top);
    for (;;) {
        __global slab_t *sp = slabptr(top);
        if (sp) {
            ulong next2 = AL(&sp->next2);
            if (ACE(&hp->top, &top, addcnt(next2, top)))
                return sp;
        } else
            return 0;
        slab_pause();
    }
}

// reset slab, called by a single workitem
NO_SANITIZE_ADDR
static void
reset_slab(__global slab_t *sp)
{
    AS(&sp->ap, (ulong)sp + SLAB_HEADER_BYTES);
    AS(&sp->rb, 0U);
}

NO_SANITIZE_ADDR
static void
poison_allocation(__global alloc_t *ap, uint sz)
{
    __global uchar *asp = (__global uchar *)MEM_TO_SHADOW((ulong)ap) + ALLOC_HEADER_BYTES / SHADOW_GRANULARITY;
    for (uint i = 0; i < (sz + SHADOW_GRANULARITY - 1) / SHADOW_GRANULARITY; ++i)
        asp[i] = kAsanHeapFreeMagic;

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

NO_SANITIZE_ADDR
static void
unpublish_allocation(__global alloc_t *ap, ulong pc)
{
    ap->pc = pc;
    poison_allocation(ap, ap->usz);
}

// Free a slab based allocation
NO_SANITIZE_ADDR
static void
slab_free(__global alloc_t *ap, ulong pc)
{
    unpublish_allocation(ap, pc);
    __global heap_t *hp = get_heap_ptr();
    __global slab_t *sp = (__global slab_t *)ap->sp;
    int go = 1;
    do {
        O0(go);
        if (go) {
            if (sp == first(sp)) {
                uint sz = __ockl_alisa_u32(ap->asz);
                uint aid = __ockl_activelane_u32();
                if (aid == 0) {
                    uint rb = AA(&sp->rb, sz) + sz;
                    if (rb == SLAB_BYTES - SLAB_HEADER_BYTES) {
                        ulong cs = AL(&hp->cs);
                        if ((ulong)sp == cs) {
                            ACE(&hp->cs, &cs, 0UL);
                        }
                        put_free_slab(hp, sp);
                    }
                }
                go = 0;
            }
        }
    } while (__ockl_wfany_i32(go));
}

// Free a non-slab allocation
NO_SANITIZE_ADDR
static void
non_slab_free(__global alloc_t *ap, ulong pc)
{
    ap->pc = pc;
    __ockl_devmem_request((ulong)ap, 0);

#if defined NON_SLAB_TRACKING
    uint aid = __ockl_activelane_u32();
    uint nactive = active_lane_count();

    if (aid == 0) {
        __global heap_t *hp = get_heap_ptr();
        AA(&hp->num_nonslab_allocations, -nactive);
    }
#endif
}

// free
USED
NO_INLINE
NO_SANITIZE_ADDR
void
__asan_free_impl(ulong aa, ulong pc)
{
    if (!aa)
        return;

    pc -= CALL_BYTES;

    uptr sa = MEM_TO_SHADOW(aa);
    s8 sb = *(__global s8*) sa;
    if (sb != 0 && ((s8)(aa & (SHADOW_GRANULARITY-1)) >= sb)) {
        REPORT_IMPL(pc, aa, 1, 1, false);
    }

    __global alloc_t *ap = (__global alloc_t *)(aa - ALLOC_HEADER_BYTES);
    if (ap->sp)
        slab_free(ap, pc);
    else
        non_slab_free(ap, pc);
}

// Non-slab based allocation (when size is above threshold)
NO_SANITIZE_ADDR
static ulong
non_slab_malloc(ulong sz, ulong pc)
{
    ulong ret = __ockl_devmem_request(0UL, sz + ALLOC_HEADER_BYTES);
    if (ret) {
#if defined NON_SLAB_TRACKING
        uint aid = __ockl_activelane_u32();
        uint nactive = active_lane_count();

        if (aid == 0) {
            __global heap_t *hp = get_heap_ptr();
            AA(&hp->num_nonslab_allocations, nactive);
        }
#endif

#if SLAB_HEADER_BYTES == 32
        __global uint *asp = (__global uint *)MEM_TO_SHADOW(ret);
        *asp = kAsanHeapLeftRedzoneMagicx4;
#else
#error unimplemented poisoning
#endif

        __global alloc_t *ap = (__global alloc_t *)ret;
        ap->magic = ALLOC_MAGIC;
        ap->sp = 0UL;
        ap->pc = pc;
        ap->asz = (uint)(sz + ALLOC_HEADER_BYTES);
        ap->usz = (uint)sz;
        ret += ALLOC_HEADER_BYTES;
    }
    return ret;
}

// Called by a single workitem
NO_SANITIZE_ADDR
static __global slab_t *
obtain_new_slab(__global heap_t *hp)
{
    ulong ret = 0;

    ulong is = AL(&hp->initial_slabs);
    ulong se = hp->initial_slabs_end;
    if (is < se) {
        is = AA(&hp->initial_slabs, SLAB_BYTES);
        if (is < se)
            ret = is;
    } else {
        ret = __ockl_devmem_request(0, SLAB_BYTES);
    }

    if (ret)
        AA(&hp->nas, 1);

    return (__global slab_t *)ret;
}

// Called by a single workitem
NO_SANITIZE_ADDR
static __global slab_t *
try_new_slab(__global heap_t *hp)
{
    ulong atime = AL(&hp->atime);
    ulong now = __ockl_steadyctr_u64();
    ulong dt = now - atime;
    if  (dt < SLAB_TICKS || !ACE(&hp->atime, &atime, now))
        return SLAB_BUSY;

    __global slab_t *sp = obtain_new_slab(hp);
    if (sp) {
        AS(&sp->next, 0UL);
        AS(&sp->next2, 0UL);
        AS(&sp->ap, (ulong)sp->space);
        AS(&sp->rb, 0U);
        AS(&sp->flags, 0U);
    }
    return sp;
}

// Called by a single workitem
NO_SANITIZE_ADDR
static void
new_slab_wait(__global heap_t *hp)
{
    ulong atime = AL(&hp->atime);
    ulong now = __ockl_steadyctr_u64();
    ulong dt = now - atime;
    if  (dt < SLAB_TICKS)
        __ockl_rtcwait_u32(SLAB_TICKS - (uint)dt);
}

// Called by a single workitem
NO_SANITIZE_ADDR
static __global slab_t *
get_current_slab(__global heap_t *hp)
{
    for (;;) {
        ulong cs = AL(&hp->cs);
        if (cs)
            return (__global slab_t *)cs;

        slab_pause();

        cs = AL(&hp->cs);
        if (cs)
            return (__global slab_t *)cs;

        slab_pause();

        cs = AL(&hp->cs);
        if (cs)
            return (__global slab_t *)cs;

        if (AL(&hp->nas) >= SLAB_RECYCLE_THRESHOLD) {
            __global slab_t *fs = get_free_slab(hp);
            if (fs) {
                reset_slab(fs);
                if (ACE(&hp->cs, &cs, (ulong)fs))
                    return fs;
                put_free_slab(hp, fs);
                return (__global slab_t *)cs;
            }
        }

        __global slab_t *ns = try_new_slab(hp);
        if ((ulong)ns > (ulong)SLAB_BUSY) {
            if (ACE(&hp->cs, &cs, (ulong)ns))
                return ns;
            put_free_slab(hp, ns);
            return (__global slab_t *)cs;
        }

        if (!ns)
            return 0;

        new_slab_wait(hp);
    }
}

NO_SANITIZE_ADDR
static void
poison_slab(__global slab_t *sp, int aid, int na)
{
    __global ulong *ssp = (__global ulong *)MEM_TO_SHADOW((ulong)sp);

    for (int i=aid; i < SLAB_BYTES / SHADOW_GRANULARITY / sizeof(ulong); i += na)
        ssp[i] = kAsanHeapLeftRedzoneMagicx8;

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

    if (!aid)
        AO(&sp->flags, 2);
}

NO_SANITIZE_ADDR
static void
poison_slab_wait(__global slab_t *sp)
{
    while (AL(&sp->flags) != 3)
        slab_pause();
}

NO_SANITIZE_ADDR
static void
unpoison_allocation(__global alloc_t *ap, uint sz)
{
    __global uchar *asp = (__global uchar *)MEM_TO_SHADOW((ulong)ap) + ALLOC_HEADER_BYTES / SHADOW_GRANULARITY;
    for (uint i = 0; i < sz / SHADOW_GRANULARITY; ++i)
        asp[i] = (uchar)0;

    if (sz % SHADOW_GRANULARITY)
        asp[sz / SHADOW_GRANULARITY] = (uchar)(sz % SHADOW_GRANULARITY);

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

NO_SANITIZE_ADDR
static void
publish_allocation(__global alloc_t *ap, ulong sp, ulong pc, uint asz, uint usz)
{
    ap->magic = ALLOC_MAGIC;
    ap->pc = pc;
    ap->sp = sp;
    ap->asz = asz;
    ap->usz = usz;

    unpoison_allocation(ap, usz);
}

// slab based malloc
NO_SANITIZE_ADDR
static ulong
slab_malloc(ulong lsz, ulong pc)
{
    __global heap_t *hp = get_heap_ptr();
    uint usz = (uint)lsz;
    uint arz = added_redzone(usz);
    uint asz = arz + ALLOC_HEADER_BYTES + round_16(usz);
    ulong ret = 0;

    int go = 1;
    do {
        if (go) {
            O0(go);
            uint aid = __ockl_activelane_u32();

            __global slab_t *cs = (__global slab_t *)0;
            if (!aid)
                cs = get_current_slab(hp);
            cs = first(cs);

            if (!cs) {
                go = 0;
                continue;
            }

            uint f = 0U;
            if (!aid) {
                f = AO(&cs->flags, 1U);
            }
            f = first(f);
            if (!f) {
                poison_slab(cs, aid, active_lane_count());
            } else if (f == 1) {
                if (!aid)
                    poison_slab_wait(cs);
            }

            uint o = __ockl_alisa_u32(asz);

            ulong p = 0UL;
            if (!aid)
                p = AA(&cs->ap, o);
            p = first(p);

            if (p + o <= (ulong)cs + SLAB_BYTES) {
                __global alloc_t *ap = (__global alloc_t *)(p + o - asz + arz);
                publish_allocation(ap, (ulong)cs, pc, asz, usz);
                ret = (ulong)ap + ALLOC_HEADER_BYTES;
                go = 0;
            } else {
                if (!__ockl_activelane_u32()) {
                    ulong e = (ulong)cs;
                    ACE(&hp->cs, &e, 0UL);
                }
                if (p + o - asz < (ulong)cs + SLAB_BYTES) {
                    uint unused = (uint)((ulong)cs + SLAB_BYTES - (p + o - asz));
                    uint rb = AA(&cs->rb, unused) + unused;

                    if (rb == SLAB_BYTES - SLAB_HEADER_BYTES)
                        put_free_slab(hp, cs);
                }
            }
        }
    } while (__ockl_wfany_i32(go));

    return ret;
}

// malloc
USED
NO_INLINE
NO_SANITIZE_ADDR
ulong
__asan_malloc_impl(ulong sz, ulong pc)
{
    pc -= CALL_BYTES;

    if (sz > SLAB_THRESHOLD)
        return non_slab_malloc(sz, pc);
    else
        return slab_malloc(sz, pc);
}

// This initialization assumes a one-workgroup grid with 256 work items,
// exacty like the non-ASAN version
NO_SANITIZE_ADDR
void
__ockl_dm_init_v1(ulong ha, ulong sa, uint hb, uint nis)
{
    uint lid = __ockl_get_local_id(0);

    __global ulong *hs = (__global ulong *)MEM_TO_SHADOW(ha);
    hs[lid+0*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+1*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+2*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+3*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+4*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+5*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+6*256] = kAsanHeapLeftRedzoneMagicx8;
    hs[lid+7*256] = kAsanHeapLeftRedzoneMagicx8;

    if (lid == 0) {
        __global heap_t *hp = (__global heap_t *)ha;
        AS(&hp->fake_next, 0UL);
        AS(&hp->fake_next2, 0UL);
        AS(&hp->head, (ulong)&hp->fake_next);
        AS(&hp->tail, (ulong)&hp->fake_next);
        AS(&hp->top, 0UL);
        AS(&hp->cs, 0UL);
        AS(&hp->initial_slabs, sa);
        hp->initial_slabs_end = sa + ((ulong)nis << 21);
        AS(&hp->nas, 0U);
#if defined NON_SLAB_TRACKING
        AS(&hp->num_nonslab_allocations, 0UL);
#endif
    }
}

NO_SANITIZE_ADDR
void
__ockl_dm_trim(int *mem)
{
}

#if defined NON_SLAB_TRACKING
// return a snapshot of the current number of nonslab allocations
// which haven't been deallocated
NO_SANITIZE_ADDR
ulong
__ockl_dm_nna(void)
{
    __global heap_t *hp = get_heap_ptr();
    return AL(&hp->num_nonslab_allocations);
}
#endif

