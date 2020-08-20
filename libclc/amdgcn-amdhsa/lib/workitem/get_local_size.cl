#include <clc/clc.h>

#if __clang_major__ >= 8
#define CONST_AS __constant
#elif __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

#if __clang_major__ >= 6
#define __dispatch_ptr __builtin_amdgcn_dispatch_ptr
#else
#define __dispatch_ptr __clc_amdgcn_dispatch_ptr
CONST_AS char * __clc_amdgcn_dispatch_ptr(void) __asm("llvm.amdgcn.dispatch.ptr");
#endif

_CLC_DEF _CLC_OVERLOAD size_t get_local_size(uint dim) {
  CONST_AS uint *ptr = (CONST_AS uint *)__dispatch_ptr();
  switch (dim) {
  case 0:
    return ptr[1] & 0xffffu;
  case 1:
    return ptr[1] >> 16;
  case 2:
    return ptr[2] & 0xffffu;
  }
  return 1;
}
