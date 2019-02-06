//===-------------------------- test_aux_runtime_op_array_new.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-exceptions

#include <iostream>
#include <cxxabi.h>

//  If the expression passed to operator new[] would result in an overflow, the
//  allocation function is not called, and a std::bad_array_new_length exception
//  is thrown instead (5.3.4p7).
bool bad_array_new_length_test() {
    try {
      // We test this directly because Clang does not currently codegen the
      // correct call to __cxa_bad_array_new_length, so this test would result
      // in passing -1 to ::operator new[], which would then throw a
      // std::bad_alloc, causing the test to fail.
      __cxxabiv1::__cxa_throw_bad_array_new_length();
    } catch ( const std::bad_array_new_length &banl ) {
      return true;
    }
    return false;
}

int main() {
    int ret_val = 0;

    if ( !bad_array_new_length_test ()) {
        std::cerr << "Bad array new length test failed!" << std::endl;
        ret_val = 1;
    }

    return ret_val;
}
