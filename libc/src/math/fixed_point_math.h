#ifndef LLVM_LIBC_SRC_MATH_FIXED_POINT_MATH_H
#define LLVM_LIBC_SRC_MATH_FIXED_POINT_MATH_H

#include <stdint.h>

namespace __llvm_libc {

// Multiply two fixed-point numbers and return the result
constexpr int32_t mulifx(int32_t a, int32_t b, int frac_bits) {
    int64_t product = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(product >> frac_bits);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_FIXED_POINT_MATH_H
