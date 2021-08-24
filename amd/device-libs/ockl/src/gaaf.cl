
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

#define AS_FLOAT(X) __builtin_astype(X, float)
#define AS_UINT(X) __builtin_astype(X, uint)

#define AC(P, E, V, O, R, S) __opencl_atomic_compare_exchange_strong(P, E, V, O, R, S)
#define AL(P, O, S) __opencl_atomic_load(P, O, S)

extern float __llvm_amdgcn_global_atomic_fadd_f32_p1f32_f32(__global float *, float) __asm("llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32");

__attribute__((target("atomic-fadd-insts"))) static void
global_atomic_fadd(__global float *p, float v)
{
    __llvm_amdgcn_global_atomic_fadd_f32_p1f32_f32(p, v);
}

static void
generic_atomic_fadd(float *p, float v)
{
    atomic_uint *t = (atomic_uint *)p;
    uint e = AL(t, memory_order_relaxed, memory_scope_device);
    while (!AC(t, &e, AS_UINT(v + AS_FLOAT(e)), memory_order_relaxed, memory_order_relaxed, memory_scope_device))
        ;
}

void
__ockl_atomic_add_noret_f32(float *p, float v)
{
    if ((__oclc_ISA_version == 9008 || __oclc_ISA_version == 9010) && !__ockl_is_local_addr(p) && !__ockl_is_private_addr(p)) {
        global_atomic_fadd((__global float *)p, v);
    } else {
        generic_atomic_fadd(p, v);
    }
}

