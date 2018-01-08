#define __CLC_DST_ADDR_SPACE local
#define __CLC_SRC_ADDR_SPACE global
#define __CLC_BODY <clc/async/async_work_group_copy.inc>
#include <clc/async/gentype.inc>
#undef __CLC_DST_ADDR_SPACE
#undef __CLC_SRC_ADDR_SPACE
#undef __CLC_BODY

#define __CLC_DST_ADDR_SPACE global
#define __CLC_SRC_ADDR_SPACE local
#define __CLC_BODY <clc/async/async_work_group_copy.inc>
#include <clc/async/gentype.inc>
#undef __CLC_DST_ADDR_SPACE
#undef __CLC_SRC_ADDR_SPACE
#undef __CLC_BODY
