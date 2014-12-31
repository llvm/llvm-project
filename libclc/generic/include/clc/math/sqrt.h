#undef sqrt
#define sqrt __clc_sqrt

#define __CLC_FUNCTION __clc_sqrt
#define __CLC_INTRINSIC "llvm.sqrt"
#include <clc/math/unary_intrin.inc>
