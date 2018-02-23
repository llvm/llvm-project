#include <clc/clc.h>

#if __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

_CLC_DEF uint get_work_dim(void)
{
	CONST_AS uint * ptr =
		(CONST_AS uint *) __builtin_amdgcn_implicitarg_ptr();
	return ptr[0];
}
