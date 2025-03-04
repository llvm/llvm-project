#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD size_t get_global_size(uint dim) {
  switch (dim) {
  case 0:
    return __builtin_amdgcn_grid_size_x();
  case 1:
    return __builtin_amdgcn_grid_size_y();
  case 2:
    return __builtin_amdgcn_grid_size_z();
  default:
    return 1;
  }
}
