#include <clc/clc.h>
#include <integer/popcount.h>

#define __CLC_FUNC popcount
#define __CLC_IMPL_FUNC __clc_native_popcount

#define __CLC_BODY "../clc_unary.inc"
#include <clc/integer/gentype.inc>
