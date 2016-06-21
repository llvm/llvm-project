
#include "llvm.h"
#include "ockl.h"

__attribute__((always_inline, const)) int
OCKL_MANGLE_I32(clz)(int i)
{
    return i ? __llvm_ctlz_i32(i) : 32;
}

