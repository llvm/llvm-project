#include <clc/clc.h>

_CLC_DEF size_t get_group_id(uint dim)
{
	switch(dim) {
	case 0: return __builtin_amdgcn_workgroup_id_x();
	case 1: return __builtin_amdgcn_workgroup_id_y();
	case 2: return __builtin_amdgcn_workgroup_id_z();
	default: return 1;
	}
}
