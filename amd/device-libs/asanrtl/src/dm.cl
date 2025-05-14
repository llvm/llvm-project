/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "asan_util.h"
#include "shadow_mapping.h"

#define OPTNONE __attribute__((optnone))

static const __constant uchar kAsanHeapLeftRedzoneMagic = (uchar)0xfa;
static const __constant uint kAsanHeapLeftRedzoneMagicx4 = 0xfafafafaU;
static const __constant ulong kAsanHeapLeftRedzoneMagicx8 = 0xfafafafafafafafaUL;
static const __constant uchar kAsanHeapFreeMagic = (uchar)0xfd;
static const __constant uchar kAsanArrayCookieMagic = (uchar)0xac;

extern ulong __ockl_devmem_request(ulong addr, ulong size);

// Whether we track non-slab allocations
#define NON_SLAB_TRACKING 1

// Whether we add ID to slabs
#define SLAB_IDENTITY 1

// Magic at beginning of allocation
#define ALLOC_MAGIC 0xfedcba1ee1abcdefUL

#define AS(P,V) __opencl_atomic_store(P, V, memory_order_relaxed, memory_scope_device)
#define AL(P) __opencl_atomic_load(P, memory_order_relaxed, memory_scope_device)
#define AA(P,V) __opencl_atomic_fetch_add(P, V, memory_order_relaxed, memory_scope_device)
#define AN(P,V) __opencl_atomic_fetch_and(P, V, memory_order_relaxed, memory_scope_device)
#define AO(P,V) __opencl_atomic_fetch_or(P, V, memory_order_relaxed, memory_scope_device)
#define ACE(P,E,V) __opencl_atomic_compare_exchange_strong(P, E, V, memory_order_relaxed, memory_order_relaxed, memory_scope_device)
#define RF() __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent", "global")
#define ARF() __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent", "global")

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
#define SLAB_BUSY ((__global slab_t *)1UL)
#define SLAB_TICKS 100000
#define SLAB_BYTES (1UL << 21)
#define SLAB_THRESHOLD (SLAB_BYTES / 64)
#define SLAB_HEADER_BYTES 32

// Assume SLAB_ALIGN so low 12 bits are already clear
#define SLAB_SHIFT 6
#define SLAB_CTR_MASK ((1UL << (SLAB_SHIFT+12)) - 1UL)

#define LINE 128
#define PAD(N,M) ulong pad##N[LINE/8 - M];

#define F_POISON_NEEDED 0x01
#define F_POISON_PENDING 0x02
#define F_UNREADY 0x04
#define F_MASK (F_POISON_NEEDED | F_POISON_PENDING | F_UNREADY)

// A slab of memory used to provide malloc returned blocks
typedef struct slab_s {
    atomic_ulong next;   // link to next slab on queue chain, must be first
    atomic_ulong ap;     // Pointer to next allocation and flags
    atomic_uint rb;      // returned bytes
    uint pad;
    atomic_ulong sid;    // slab ID
    ulong space[(SLAB_BYTES-SLAB_HEADER_BYTES)/8];  // Space for allocations.  Must  be aligned 16
} slab_t;

// A LIFO for storing available slabs
typedef struct lifo_s {
    atomic_ulong top;
    PAD(0,1);
} lifo_t;

// Number of LIFO we use, need to size to keep heap_s under 128K
// Current initialization must change if this exceeds 256
#define NLA 256
#define LP(H,I) (H->la + (I) % NLA)

// State for mechanism
typedef struct heap_s {
    atomic_ulong cs;                      // current slab pointer
    PAD(0,1);
    atomic_ulong atime;                   // Time most recent allocation started
    PAD(1,1);
    atomic_ulong rid;                     // Next read index
    PAD(2,1);
    atomic_ulong wid;                     // Next write index
    PAD(3,1);
    atomic_ulong initial_slabs;           // pointer to next preallocated slab
    ulong initial_slabs_end;              // pointer to end of preallocated slabs
    PAD(4,2);
#if defined NON_SLAB_TRACKING
    atomic_ulong num_nonslab_allocations; // Count of number of non-slab allocations that have not been freed
    PAD(5,1);
#endif
#if defined SLAB_IDENTITY
    atomic_ulong num_slab_allocations;    // Count of total slabs allocated
    PAD(6,1);
#endif
    lifo_t la[NLA];                       // Storage for available slabs
} heap_t;

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
active_lane_count(void)
{
    return __builtin_popcountl(__builtin_amdgcn_ballot_w64(true));
}

static ulong
round_16(ulong n)
{
    return ((n + 15) >> 4) << 4;
}

static ulong
addcnt(ulong p, ulong c)
{
    return (p << SLAB_SHIFT) | ((c + 1UL) & SLAB_CTR_MASK);
}

static __global slab_t *
slabptr(ulong p)
{
    return (__global slab_t *)((p & ~SLAB_CTR_MASK) >> SLAB_SHIFT);
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
    __builtin_amdgcn_s_sleep(9);
}


// Intended to be called from only one lane of a wave
OPTNONE
NO_SANITIZE_ADDR
static void
put_free_slab(__global heap_t *hp, __global slab_t *sp)
{
    __global lifo_t *lp = LP(hp, AA(&hp->wid, 1UL));

    for (;;) {
        ulong top = AL(&lp->top);
        AS(&sp->next, (ulong)slabptr(top));
        if (ACE(&lp->top, &top, addcnt((ulong)sp, top))) {
            return;
        }
        slab_pause();
    }
}

// Intended to be called from only one lane of a wave
NO_SANITIZE_ADDR
static __global slab_t *
get_free_slab(__global heap_t *hp)
{
    if (AL(&hp->rid) >= AL(&hp->wid))
        return 0;

    __global lifo_t *lp = LP(hp, AA(&hp->rid, 1UL));

    for (;;) {
        ulong top = AL(&lp->top);
        __global slab_t *sp = slabptr(top);
        if (sp) {
            ulong next = AL(&sp->next);
            if (ACE(&lp->top, &top, addcnt(next, top)))
                return sp;
        } else {
            return 0;
        }
        slab_pause();
    }

}

NO_SANITIZE_ADDR
static void
ready_slab(__global slab_t *sp)
{
    AS(&sp->rb, 0U);
    if (!(AL(&sp->ap) & (ulong)(F_POISON_PENDING | F_POISON_NEEDED))) {
        AS(&sp->ap, (ulong)sp + SLAB_HEADER_BYTES);
    } else {
        AN(&sp->ap, ~(ulong)F_UNREADY);
    }
}

NO_SANITIZE_ADDR
static void
unpublish_allocation(__global alloc_t *ap, ulong pc)
{
     uint arz = ap->asz - ALLOC_HEADER_BYTES - round_16(ap->usz);
    __global uchar *s = (__global uchar *)MEM_TO_SHADOW((ulong)ap - arz);
    __builtin_memset(s, kAsanHeapFreeMagic, ap->asz / SHADOW_GRANULARITY);
    ap->pc = pc;
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
        if (go) {
            if (sp == first(sp)) {
                uint sz = __ockl_alisa_u32(ap->asz);
                uint aid = __ockl_activelane_u32();
                if (aid == 0) {
                    uint rb = AA(&sp->rb, sz) + sz;
                    if (rb == SLAB_BYTES - SLAB_HEADER_BYTES) {
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

    ARF();

    uptr sa = MEM_TO_SHADOW(aa);
    s8 sb = *(__global s8*) sa;
    if (sb != 0 && sb != (s8)kAsanArrayCookieMagic && ((s8)(aa & (SHADOW_GRANULARITY-1)) >= sb)) {
        REPORT_IMPL(pc, aa, 1, 1, false);
    }

    __global alloc_t *ap = (__global alloc_t *)(aa - ALLOC_HEADER_BYTES);
    if (ap->sp)
        slab_free(ap, pc);
    else
        non_slab_free(ap, pc);

    ARF();
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
        AS(&sp->rb, 0U);
        AS(&sp->ap, (ulong)sp + (ulong)SLAB_HEADER_BYTES + (ulong)(F_UNREADY | F_POISON_PENDING | F_POISON_NEEDED));
#if defined SLAB_IDENTITY
        AS(&sp->sid, AA(&hp->num_slab_allocations, 1UL));
#else
        AS(&sp->sid, 0UL);
#endif
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
OPTNONE
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

        __global slab_t *fs = get_free_slab(hp);
        if (fs) {
            if (ACE(&hp->cs, &cs, (ulong)fs)) {
                ready_slab(fs);
                return fs;
            }
            put_free_slab(hp, fs);
            continue;
        }

        __global slab_t *ns = try_new_slab(hp);
        if ((ulong)ns > (ulong)SLAB_BUSY) {
            if (ACE(&hp->cs, &cs, (ulong)ns)) {
                AN(&ns->ap, ~(ulong)F_UNREADY);
                return ns;
            }
            put_free_slab(hp, ns);
            continue;
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
    RF();

    if (!aid)
        AN(&sp->ap, ~(ulong)F_POISON_PENDING);
}

NO_SANITIZE_ADDR
static ulong
publish_allocation(ulong ap, ulong sp, ulong pc, uint asz, uint arz, uint usz)
{
    __global uchar *s = (__global uchar *)MEM_TO_SHADOW(ap);

    __builtin_memset(s, kAsanHeapLeftRedzoneMagic, (arz + ALLOC_HEADER_BYTES) / SHADOW_GRANULARITY);

    s += (arz + ALLOC_HEADER_BYTES) / SHADOW_GRANULARITY;
    __builtin_memset(s, 0, usz / SHADOW_GRANULARITY);
    if (usz % SHADOW_GRANULARITY)
        s[usz / SHADOW_GRANULARITY] = (uchar)(usz % SHADOW_GRANULARITY);

    __global alloc_t *a = (__global alloc_t *)(ap + arz);
    a->magic = ALLOC_MAGIC;
    a->sp = sp;
    a->pc = pc;
    a->asz = asz;
    a->usz = usz;

    return ap + arz + ALLOC_HEADER_BYTES;
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
            uint aid = __ockl_activelane_u32();

            __global slab_t *cs = (__global slab_t *)0;
            if (!aid)
                cs = get_current_slab(hp);
            cs = first(cs);

            if (!cs) {
                go = 0;
                continue;
            }

            ulong o = (ulong)__ockl_alisa_u32(asz);

            ulong ap = 0;
            if (!aid)
                ap = AL(&cs->ap);
            ap = first(ap);

            if (ap & (ulong)F_MASK) {
                ulong p = 0;
                if (!aid)
                    p = AN(&cs->ap, ~(ulong)F_POISON_NEEDED);
                p = first(p);

                if (p & (ulong)F_POISON_NEEDED)
                    poison_slab(cs, aid, active_lane_count());
                else
                    slab_pause();
            } else {
                ulong p = 0;
                if (!aid)
                    p = AA(&cs->ap, o);
                p = first(p);

                if (!(p & (ulong)F_MASK)) {
                    if (p + o <= (ulong)cs + SLAB_BYTES) {
                        ret = publish_allocation(p + o - asz, (ulong)cs, pc, asz, arz, usz);
                        go = 0;
                    } else {
                        if (!__ockl_activelane_u32()) {
                            ulong e = (ulong)cs;
                            ACE(&hp->cs, &e, 0UL);
                            AO(&cs->ap, (ulong)F_UNREADY);
                        }
                        if (p + o - asz < (ulong)cs + SLAB_BYTES) {
                            uint unused = (uint)((ulong)cs + SLAB_BYTES - (p + o - asz));
                            uint rb = AA(&cs->rb, unused) + unused;
                            if (rb == SLAB_BYTES - SLAB_HEADER_BYTES) {
                                put_free_slab(hp, cs);
                            }
                        }
                    }
                } else
                    slab_pause();
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

    ARF();

    ulong ret;
    if (sz > SLAB_THRESHOLD)
        ret = non_slab_malloc(sz, pc);
    else
        ret = slab_malloc(sz, pc);

    ARF();

    return ret;
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

    __global heap_t *hp = (__global heap_t *)ha;

    if (!lid) {
        AS(&hp->cs, 0UL);
        AS(&hp->atime, 0UL);
        AS(&hp->rid, 0UL);
        AS(&hp->wid, 0UL);
        AS(&hp->initial_slabs, sa);
        hp->initial_slabs_end = sa + ((ulong)nis << 21);
#if defined NON_SLAB_TRACKING
        AS(&hp->num_nonslab_allocations, 0UL);
#endif
#if defined SLAB_IDENTITY
        AS(&hp->num_slab_allocations, 0UL);
#endif
    }

    if (lid < NLA) {
        __global lifo_t *lp = LP(hp, lid);
        AS(&lp->top, 0UL);
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

