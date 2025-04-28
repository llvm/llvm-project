#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD size_t get_group_id(uint dim) {
  switch (dim) {
  case 0:  return __nvvm_read_ptx_sreg_ctaid_x();
  case 1:  return __nvvm_read_ptx_sreg_ctaid_y();
  case 2:  return __nvvm_read_ptx_sreg_ctaid_z();
  default: return 0;
  }
}
