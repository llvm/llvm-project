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
CONST_AS uchar * __clc_amdgcn_dispatch_ptr(void) __asm("llvm.amdgcn.dispatch.ptr");
#endif

_CLC_DEF size_t get_global_size(uint dim)
{
	CONST_AS uint * ptr = (CONST_AS uint *) __dispatch_ptr();
	if (dim < 3)
		return ptr[3 + dim];
	return 1;
}
