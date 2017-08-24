#undef ceil
#define ceil __clc_ceil

#define __CLC_FUNCTION __clc_ceil
#define __CLC_INTRINSIC "llvm.ceil"
#include <clc/math/unary_intrin.inc>
