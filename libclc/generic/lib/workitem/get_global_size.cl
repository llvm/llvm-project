#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD size_t get_global_size(uint dim) {
  return get_num_groups(dim)*get_local_size(dim);
}
