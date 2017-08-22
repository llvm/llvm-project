#include <clc/clc.h>

_CLC_DEF uint get_group_id(uint dim)
{
	switch(dim) {
	case 0: return __builtin_r600_read_tgid_x();
	case 1: return __builtin_r600_read_tgid_y();
	case 2: return __builtin_r600_read_tgid_z();
	default: return 1;
	}
}
