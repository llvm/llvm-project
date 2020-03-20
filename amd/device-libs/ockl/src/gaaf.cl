
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

extern void __llvm_amdgcn_global_atomic_fadd_p1f32_f32(__global float *, float) __asm("llvm.amdgcn.global.atomic.fadd.p1f32.f32");

void
__ockl_atomic_add_noret_f32(float *p, float v)
{
    if (__oclc_ISA_version == 9008 && !__ockl_is_local_addr(p) && !__ockl_is_private_addr(p)) {
        __llvm_amdgcn_global_atomic_fadd_p1f32_f32((__global float *)p, v);
    } else {
        atomic_uint *t = (atomic_uint *)p;
        uint e = AL(t, memory_order_relaxed, memory_scope_device);
        while (!AC(t, &e, AS_UINT(v + AS_FLOAT(e)), memory_order_relaxed, memory_order_relaxed, memory_scope_device))
            ;
    }
}

