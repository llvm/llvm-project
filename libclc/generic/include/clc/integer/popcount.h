#undef popcount
#define popcount __clc_popcount

#define __CLC_FUNCTION __clc_popcount
#define __CLC_INTRINSIC "llvm.ctpop"
#include <clc/integer/unary_intrin.inc>
