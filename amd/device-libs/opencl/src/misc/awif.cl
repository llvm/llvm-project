/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#pragma OPENCL EXTENSIOM cl_khr_subgroups : enable

__attribute__((overloadable, always_inline)) void
mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_acq_rel, memory_scope_work_group);
}

__attribute__((overloadable, always_inline)) void
read_mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_acquire, memory_scope_work_group);
}

__attribute__((overloadable, always_inline)) void
write_mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_release, memory_scope_work_group);
}

#if !defined LOW_LEVEL_APPROACH
__attribute__((overloadable, always_inline)) void
atomic_work_item_fence(cl_mem_fence_flags flags, memory_order order, memory_scope scope)
{
    // We're tying global-happens-before and local-happens-before together as does HSA
    if (order != memory_order_relaxed) {
        switch (scope) {
        case memory_scope_work_item:
            break;
        case memory_scope_sub_group:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_sg(); break;
            case memory_order_release: __llvm_fence_rel_sg(); break;
            case memory_order_acq_rel: __llvm_fence_ar_sg(); break;
            case memory_order_seq_cst: __llvm_fence_sc_sg(); break;
            }
            break;
        case memory_scope_work_group:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_wg(); break;
            case memory_order_release: __llvm_fence_rel_wg(); break;
            case memory_order_acq_rel: __llvm_fence_ar_wg(); break;
            case memory_order_seq_cst: __llvm_fence_sc_wg(); break;
            }
            break;
        case memory_scope_device:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_dev(); break;
            case memory_order_release: __llvm_fence_rel_dev(); break;
            case memory_order_acq_rel: __llvm_fence_ar_dev(); break;
            case memory_order_seq_cst: __llvm_fence_sc_dev(); break;
            }
            break;
        case memory_scope_all_svm_devices:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_sys(); break;
            case memory_order_release: __llvm_fence_rel_sys(); break;
            case memory_order_acq_rel: __llvm_fence_ar_sys(); break;
            case memory_order_seq_cst: __llvm_fence_sc_sys(); break;
            }
            break;
        }
    }
}
#else
// LGKMC (LDS, GDS, Konstant, Message) is 4 bits
// EXPC (Export) is 3 bits
// VMC (VMem) is 4 bits
#define LGKMC_MAX 0xf
#define EXPC_MAX 0x7
#define VMC_MAX 0xf
#define WAITCNT_IMM(LGKMC, EXPC, VMC) ((LGKMC << 8) | (EXPC << 4) | VMC)

__attribute__((overloadable, always_inline)) void
atomic_work_item_fence(cl_mem_fence_flags flags, memory_order order, memory_scope scope)
{
    if (order != memory_order_relaxed) {
        // Strip CLK_IMAGE_MEM_FENCE
        flags &= CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE;

        if (flags == CLK_LOCAL_MEM_FENCE) {
            __llvm_amdgcn_s_waitcnt(WAITCNT_IMM(0, EXPC_MAX, VMC_MAX));
        } else if (flags == CLK_GLOBAL_MEM_FENCE) {
            if (order != memory_order_acquire) {
                __llvm_amdgcn_s_waitcnt(WAITCNT_IMM(LGKMC_MAX, EXPC_MAX, 0));
                __llvm_amdgcn_s_dcache_wb();
            }

            if ((scope == memory_scope_device) | (scope == memory_scope_all_svm_devices)) {
                if (order != memory_order_release) {
                    __llvm_amdcgn_buffer_wbinvl1_vol();
                    __llvm_amdgcn_s_dcache_inv_vol();
                }
            }
        } else if (flags == (CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE)) {
            __llvm_amdgcn_s_waitcnt(order == memory_order_acquire ?
                                    WAITCNT_IMM(0, EXPC_MAX, VMC_MAX) :
                                    WAITCNT_IMM(0, EXPC_MAX, 0));
            if (order != memory_order_acquire)
                __llvm_amdgcn_s_dcache_wb();

            if ((scope == memory_scope_device) | (scope == memory_scope_all_svm_devices)) {
                if (order != memory_order_release) {
                    __llvm_amdcgn_buffer_wbinvl1_vol();
                    __llvm_amdgcn_s_dcache_inv_vol();
                }
            }
        }
    }
}
#endif // LOW_LEVEL_APPROACH

