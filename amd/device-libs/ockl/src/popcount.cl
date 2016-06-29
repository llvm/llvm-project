
#include "llvm.h"
#include "ockl.h"

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(popcount)(uint i)
{
    return (uint)__llvm_ctpop_i32((int)i);
}

