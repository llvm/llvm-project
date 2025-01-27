/* Duplciate these so we don't have to distribute utils.h */
#define __CLC_CONCAT(x, y) x ## y
#define __CLC_XCONCAT(x, y) __CLC_CONCAT(x, y)

#define __CLC_SELECT_FN select

#define __CLC_BODY <clc/relational/clc_select_decl.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select_decl.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_SELECT_FN
#undef __CLC_CONCAT
#undef __CLC_XCONCAT
