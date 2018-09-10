#undef rint
#define rint __clc_rint

#define __CLC_FUNCTION __clc_rint
#define __CLC_INTRINSIC "llvm.rint"
#include <clc/math/unary_intrin.inc>
