#undef floor
#define floor __clc_floor

#define __CLC_FUNCTION __clc_floor
#define __CLC_INTRINSIC "llvm.floor"
#include <clc/math/unary_intrin.inc>
