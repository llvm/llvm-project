#include <clc/clc.h>

_CLC_DEF size_t get_num_groups(uint dim) {
  switch (dim) {
  case 0:  return __nvvm_read_ptx_sreg_nctaid_x();
  case 1:  return __nvvm_read_ptx_sreg_nctaid_y();
  case 2:  return __nvvm_read_ptx_sreg_nctaid_z();
  default: return 0;
  }
}
