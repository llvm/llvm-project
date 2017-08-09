#undef fma
#define fma __clc_fma

#define __CLC_FUNCTION __clc_fma
#define __CLC_INTRINSIC "llvm.fma"
#include <clc/math/ternary_intrin.inc>
