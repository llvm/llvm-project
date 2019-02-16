#undef trunc
#define trunc __clc_trunc

#define __CLC_FUNCTION __clc_trunc
#define __CLC_INTRINSIC "llvm.trunc"
#include <clc/math/unary_intrin.inc>

#undef __CLC_FUNCTION
#undef __CLC_INTRINSIC
