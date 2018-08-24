
#include <clc/clc.h>

_CLC_DEF size_t get_num_groups(uint dim) {
  size_t global_size = get_global_size(dim);
  size_t local_size = get_local_size(dim);
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0) {
    num_groups++;
  }
  return num_groups;
}
