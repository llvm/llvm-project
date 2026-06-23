//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <utility>

// expected-error@+1 {{pointer to subobject of string literal is not allowed in a template argument}}
std::constant_wrapper<"hello"> string_literal;

// expected-error-re@*:* {{static assertion failed{{.*}}the second template parameter of std::constant_wrapper must be its value_type}}
std::constant_wrapper<1, float> wrong_type1; // expected-note {{in instantiation of template class}}
// expected-error-re@*:* {{static assertion failed{{.*}}the second template parameter of std::constant_wrapper must be its value_type}}
std::constant_wrapper<1.0, int> wrong_type2; // expected-note {{in instantiation of template class}}
// expected-error-re@*:* {{static assertion failed{{.*}}the second template parameter of std::constant_wrapper must be its value_type}}
std::constant_wrapper<1, const int> wrong_type3; // expected-note {{in instantiation of template class}}
// expected-error-re@*:* {{static assertion failed{{.*}}the second template parameter of std::constant_wrapper must be its value_type}}
std::constant_wrapper<1, const int&> wrong_type4; // expected-note {{in instantiation of template class}}
