#include <clc/clc.h>

_CLC_DEF size_t get_global_offset(uint dim)
{
	__attribute__((address_space(2))) uint * ptr =
		(__attribute__((address_space(2))) uint *)
		__builtin_amdgcn_implicitarg_ptr();
	if (dim < 3)
		return ptr[dim + 1];
	return 0;
}
