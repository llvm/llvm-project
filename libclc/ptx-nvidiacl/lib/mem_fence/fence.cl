#include <clc/clc.h>

_CLC_DEF void mem_fence(cl_mem_fence_flags flags) {
   if (flags & (CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE))
     __nvvm_membar_cta();
}

// We do not have separate mechanism for read and write fences.
_CLC_DEF void read_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}

_CLC_DEF void write_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}
