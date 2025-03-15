#ifndef LLVM_LIBC_SRC_MATH_FXDIVI_H
#define LLVM_LIBC_SRC_MATH_FXDIVI_H

#include <stdint.h>

namespace __llvm_libc {

// Fixed-point division function
int32_t fxdivi(int32_t a, int32_t b, int frac_bits);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_FXDIVI_H
