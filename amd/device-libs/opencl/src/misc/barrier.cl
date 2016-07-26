
#include "llvm.h"

__attribute__((always_inline, overloadable)) void
barrier(cl_mem_fence_flags flags)
{
    // waitcnt operand bits: [0..3]=VM_CNT, [4..6]=EXP_CNT (Export), [8..11]=LGKM_CNT (LDS, GDS, Konstant, Message)
    if (flags == CLK_LOCAL_MEM_FENCE) {
        __llvm_amdgcn_s_waitcnt(0x07f);
    } else if (flags == CLK_GLOBAL_MEM_FENCE) {
        __llvm_amdgcn_s_waitcnt(0x3f0);
        __llvm_amdcgn_buffer_wbinvl1_vol();
        __llvm_amdgcn_s_dcache_wb();
        __llvm_amdgcn_s_dcache_inv_vol();
    } else if (flags == (CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE)) {
        __llvm_amdgcn_s_waitcnt(0x0f0);
        __llvm_amdcgn_buffer_wbinvl1_vol();
        __llvm_amdgcn_s_dcache_wb_vol();
        __llvm_amdgcn_s_dcache_inv_vol();
    }
    __llvm_amdgcn_s_barrier();
}

