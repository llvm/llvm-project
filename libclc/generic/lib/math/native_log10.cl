#include <clc/clc.h>

#define __CLC_FUNCTION __clc_native_log10
#define __CLC_INTRINSIC "llvm.log10"
#undef cl_khr_fp64
#include <clc/math/unary_intrin.inc>

#define __CLC_BODY <native_log10.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
