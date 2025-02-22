#include <clc/internal/clc.h>
#include <clc/utils.h>

#define __CLC_SELECT_FN __clc_select
#define __CLC_SELECT_DEF(x, y, z) return z ? y : x

#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/integer/gentype.inc>
