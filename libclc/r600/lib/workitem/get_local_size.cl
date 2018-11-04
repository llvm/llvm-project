#include <clc/clc.h>

uint __clc_r600_get_local_size_x(void) __asm("llvm.r600.read.local.size.x");
uint __clc_r600_get_local_size_y(void) __asm("llvm.r600.read.local.size.y");
uint __clc_r600_get_local_size_z(void) __asm("llvm.r600.read.local.size.z");

_CLC_DEF size_t get_local_size(uint dim)
{
	switch (dim) {
	case 0: return __clc_r600_get_local_size_x();
	case 1: return __clc_r600_get_local_size_y();
	case 2: return __clc_r600_get_local_size_z();
	default: return 1;
	}
}
