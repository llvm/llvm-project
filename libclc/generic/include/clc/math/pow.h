#undef pow
#define pow __clc_pow

#define __CLC_FUNCTION __clc_pow
#define __CLC_INTRINSIC "llvm.pow"
#include <clc/math/binary_intrin.inc>
