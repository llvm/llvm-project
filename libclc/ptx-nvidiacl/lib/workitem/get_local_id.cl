#include <clc/clc.h>

_CLC_DEF size_t get_local_id(uint dim) {
  switch (dim) {
  case 0:  return __builtin_ptx_read_tid_x();
  case 1:  return __builtin_ptx_read_tid_y();
  case 2:  return __builtin_ptx_read_tid_z();
  default: return 0;
  }
}
