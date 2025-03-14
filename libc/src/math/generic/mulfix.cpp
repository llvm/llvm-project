#include "src/math/mulifx.h"

namespace __llvm_libc {

// Multiply two fixed-point numbers and return the result
int32_t mulifx(int32_t a, int32_t b, int frac_bits) {
    int64_t product = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(product >> frac_bits);
}

} // namespace __llvm_libc
