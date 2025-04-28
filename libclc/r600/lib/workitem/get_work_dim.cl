#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD uint get_work_dim(void) {
  __attribute__((address_space(7))) uint *ptr =
      (__attribute__((address_space(7)))
       uint *)__builtin_r600_implicitarg_ptr();
  return ptr[0];
}
