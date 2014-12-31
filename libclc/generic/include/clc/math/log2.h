#undef log2
#define log2 __clc_log2

#define __CLC_FUNCTION __clc_log2
#define __CLC_INTRINSIC "llvm.log2"
#include <clc/math/unary_intrin.inc>
