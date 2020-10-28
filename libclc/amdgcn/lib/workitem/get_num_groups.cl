#include <clc/clc.h>

uint __clc_amdgcn_get_num_groups_x(void) __asm("llvm.r600.read.ngroups.x");
uint __clc_amdgcn_get_num_groups_y(void) __asm("llvm.r600.read.ngroups.y");
uint __clc_amdgcn_get_num_groups_z(void) __asm("llvm.r600.read.ngroups.z");

_CLC_DEF _CLC_OVERLOAD size_t get_num_groups(uint dim) {
  switch (dim) {
  case 0:
    return __clc_amdgcn_get_num_groups_x();
  case 1:
    return __clc_amdgcn_get_num_groups_y();
  case 2:
    return __clc_amdgcn_get_num_groups_z();
  default:
    return 1;
  }
}
