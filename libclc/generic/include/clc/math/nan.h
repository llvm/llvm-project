#define __CLC_CONCAT(x, y) x ## y
#define __CLC_XCONCAT(x, y) __CLC_CONCAT(x, y)

#define __CLC_BODY <clc/math/nan.inc>
#include <clc/math/gentype.inc>

#undef __CLC_XCONCAT
#undef __CLC_CONCAT
