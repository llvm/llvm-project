#include <clc/clc.h>

void __clc_amdgcn_s_waitcnt(unsigned flags);

// s_waitcnt takes 16bit argument with a combined number of maximum allowed
// pending operations:
// [12:8] LGKM -- LDS, GDS, Konstant (SMRD), Messages
// [7] -- undefined
// [6:4] -- exports, GDS, and mem write
// [3:0] -- vector memory operations

// Newer clang supports __builtin_amdgcn_s_waitcnt
#if __clang_major__ >= 5
#  define __waitcnt(x) __builtin_amdgcn_s_waitcnt(x)
#else
#  define __waitcnt(x) __clc_amdgcn_s_waitcnt(x)
_CLC_DEF void __clc_amdgcn_s_waitcnt(unsigned)  __asm("llvm.amdgcn.s.waitcnt");
#endif

_CLC_DEF _CLC_OVERLOAD void mem_fence(cl_mem_fence_flags flags) {
  if (flags & CLK_GLOBAL_MEM_FENCE) {
    // scalar loads are counted with LGKM but we don't know whether
    // the compiler turned any loads to scalar
    __waitcnt(0);
  } else if (flags & CLK_LOCAL_MEM_FENCE)
    __waitcnt(0xff); // LGKM is [12:8]
}
#undef __waitcnt

// We don't have separate mechanism for read and write fences
_CLC_DEF _CLC_OVERLOAD void read_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}

_CLC_DEF _CLC_OVERLOAD void write_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}
