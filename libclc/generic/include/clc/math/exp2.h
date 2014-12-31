#undef exp2
#define exp2 __clc_exp2

#define __CLC_FUNCTION __clc_exp2
#define __CLC_INTRINSIC "llvm.exp2"
#include <clc/math/unary_intrin.inc>
