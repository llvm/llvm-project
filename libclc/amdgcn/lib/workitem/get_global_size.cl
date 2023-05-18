#include <clc/clc.h>

uint __clc_amdgcn_get_global_size_x(void) __asm("llvm.r600.read.global.size.x");
uint __clc_amdgcn_get_global_size_y(void) __asm("llvm.r600.read.global.size.y");
uint __clc_amdgcn_get_global_size_z(void) __asm("llvm.r600.read.global.size.z");

_CLC_DEF _CLC_OVERLOAD size_t get_global_size(uint dim) {
  switch (dim) {
  case 0:
    return __clc_amdgcn_get_global_size_x();
  case 1:
    return __clc_amdgcn_get_global_size_y();
  case 2:
    return __clc_amdgcn_get_global_size_z();
  default:
    return 1;
  }
}
