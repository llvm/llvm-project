#include <clc/clc.h>

_CLC_DEF void barrier(cl_mem_fence_flags flags)
{
	mem_fence(flags);
	__builtin_amdgcn_s_barrier();
}
