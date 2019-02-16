#undef round
#define round __clc_round

#define __CLC_FUNCTION __clc_round
#define __CLC_INTRINSIC "llvm.round"
#include <clc/math/unary_intrin.inc>

#undef __CLC_FUNCTION
#undef __CLC_INTRINSIC
