#include <clc/clc.h>

_CLC_DEF uint get_local_id(uint dim)
{
	switch(dim) {
	case 0: return __builtin_r600_read_tidig_x();
	case 1: return __builtin_r600_read_tidig_y();
	case 2: return __builtin_r600_read_tidig_z();
	default: return 1;
	}
}
