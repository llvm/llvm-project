#include "rbits.h"

#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(fract, rbits, (int_r_t x)) {
    return fixed_point::fxbits<fract, int_r_t>(x);
}

}
