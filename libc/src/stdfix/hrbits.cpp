#include "hrbits.h" 
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(short fract, hrbits, (int_hr_t x)) {
    return fixed_point::bits(x);
}

}
