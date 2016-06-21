
#include "llvm.h"
#include "ockl.h"

__attribute__((always_inline, const)) int
OCKL_MANGLE_I32(popcount)(int i)
{
    return __llvm_ctpop_i32(i);
}

