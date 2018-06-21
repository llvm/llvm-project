#include <clc/clc.h>

#if __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

_CLC_DEF size_t get_local_size(uint dim)
{
	CONST_AS uint * ptr =
		(CONST_AS uint *) __builtin_amdgcn_dispatch_ptr();
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
