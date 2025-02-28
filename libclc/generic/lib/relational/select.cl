#include <clc/clc.h>
#include <clc/relational/clc_select.h>
#include <clc/utils.h>

#define __CLC_SELECT_FN select
#define __CLC_SELECT_DEF(x, y, z) return __clc_select(x, y, z)

#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/integer/gentype.inc>
