
#include "llvm.h"
#include "ockl.h"

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(clz)(uint i)
{
    uint r = (uint)__llvm_ctlz_i32((int)i);
    return i ? r : 32u;
}

