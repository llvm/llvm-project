
#include <clc/clc.h>

_CLC_DEF int __clc_clk_local_mem_fence() {
  return CLK_LOCAL_MEM_FENCE;
}

_CLC_DEF int __clc_clk_global_mem_fence() {
  return CLK_GLOBAL_MEM_FENCE;
}
