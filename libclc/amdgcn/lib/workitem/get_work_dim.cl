#include <clc/clc.h>

_CLC_DEF uint get_work_dim()
{
	__attribute__((address_space(2))) uint * ptr =
		(__attribute__((address_space(2))) uint *)
		__builtin_amdgcn_implicitarg_ptr();
	return ptr[0];
}
