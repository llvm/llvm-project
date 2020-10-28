#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD uint get_global_offset(uint dim) {
  __attribute__((address_space(7))) uint *ptr =
      (__attribute__((address_space(7)))
       uint *)__builtin_r600_implicitarg_ptr();
  if (dim < 3)
    return ptr[dim + 1];
  return 0;
}
