#undef fabs
#define fabs __clc_fabs

#define __CLC_FUNCTION __clc_fabs
#define __CLC_INTRINSIC "llvm.fabs"
#include <clc/math/unary_intrin.inc>
