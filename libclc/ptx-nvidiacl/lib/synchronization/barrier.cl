#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  __syncthreads();
}
