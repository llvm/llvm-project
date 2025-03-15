#include "src/math/fxdivi.h"
#include <stdint.h>

namespace __llvm_libc {

// Fixed-point division: a / b with frac_bits precision
int32_t fxdivi(int32_t a, int32_t b, int frac_bits) {
    if (b == 0) {
        // Handle division by zero case (return max value or another error handling strategy)
        return (a >= 0) ? INT32_MAX : INT32_MIN;
    }
    int64_t dividend = static_cast<int64_t>(a) << frac_bits;
    return static_cast<int32_t>(dividend / b);
}

} // namespace __llvm_libc
