#include "clc/math/clc_log2.h"
#include "clc/math/clc_log2_fast.h"

#define __CLC_FLOAT_ONLY
#define __CLC_MIN_VECSIZE 1
#define __CLC_FUNCTION __clc_log2_fast
#define __CLC_IMPL_FUNCTION(x) __builtin_amdgcn_logf(x)
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_FLOAT_ONLY

#define __CLC_HALF_ONLY
#define __CLC_IMPL_FUNCTION(x) __clc_log2
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_HALF_ONLY

#define __CLC_DOUBLE_ONLY
#define __CLC_IMPL_FUNCTION(x) __clc_log2
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_DOUBLE_ONLY
