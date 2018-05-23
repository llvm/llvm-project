//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// <type_traits>
//
// Test that is_floating_point<T>::value is true when T=__fp16 or T=_Float16.

#include <type_traits>

int main() {
  static_assert(std::is_floating_point<__fp16>::value, "");
#ifdef __FLT16_MANT_DIG__
  static_assert(std::is_floating_point<_Float16>::value, "");
#endif
  return 0;
}
