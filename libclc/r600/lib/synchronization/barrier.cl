#include <clc/clc.h>

_CLC_DEF void __clc_r600_barrier(void) __asm("llvm.r600.group.barrier");

_CLC_DEF void barrier(uint flags)
{
  // We should call mem_fence here, but that is not implemented for r600 yet
  __clc_r600_barrier();
}
