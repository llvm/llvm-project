#include <clc/clc.h>

_CLC_DEF void barrier(cl_mem_fence_flags flags) {
  __syncthreads();
}

