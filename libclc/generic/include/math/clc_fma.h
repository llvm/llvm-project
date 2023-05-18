#define __CLC_FUNCTION __clc_fma
#define __CLC_INTRINSIC "llvm.fma"
#include "math/ternary_intrin.inc"

#define __FLOAT_ONLY
#define __CLC_FUNCTION __clc_sw_fma
#define __CLC_BODY <clc/math/ternary_decl.inc>
#include <clc/math/gentype.inc>
#undef __CLC_BODY
#undef __CLC_FUNCTION
#undef __FLOAT_ONLY
