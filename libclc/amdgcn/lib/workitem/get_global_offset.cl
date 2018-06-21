#include <clc/clc.h>

#if __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

_CLC_DEF size_t get_global_offset(uint dim)
{
	CONST_AS uint * ptr =
		(CONST_AS uint *) __builtin_amdgcn_implicitarg_ptr();
	if (dim < 3)
		return ptr[dim + 1];
	return 0;
}
