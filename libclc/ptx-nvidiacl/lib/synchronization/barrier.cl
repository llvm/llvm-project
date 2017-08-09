#include <clc/clc.h>

_CLC_DEF void barrier(cl_mem_fence_flags flags) {
  if (flags & CLK_LOCAL_MEM_FENCE) {
    __syncthreads();
  }
}

