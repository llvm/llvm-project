
#include "llvm.h"
#include "ockl.h"

__attribute__((always_inline, const)) int
OCKL_MANGLE_U32(ctz)(int i)
{
    return i ? __llvm_cttz_i32(i) : 32;
}

