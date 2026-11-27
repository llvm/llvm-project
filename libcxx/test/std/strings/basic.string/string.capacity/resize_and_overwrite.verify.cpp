//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <string>

// template<class Operation>
// void resize_and_overwrite(size_type n, Operation op)

// Verify that the operation's return type must be integer-like

#include <string>

void test_bool_return_type() {
  std::string s;
  s.resize_and_overwrite(10, [](char*, std::size_t) {
    return true; // expected-error-re@*:* {{{{(static_assertion|static assertion)}}{{.*}}integer-like}}
  });
}

void test_pointer_return_type() {
  std::string s;
  s.resize_and_overwrite(10, [](char* p, std::size_t) {
    return p; // expected-error-re@*:* {{{{(static_assertion|static assertion)}}{{.*}}integer-like}}
              // expected-error@*:* {{cannot initialize}}
  });
}

void test_float_return_type() {
  std::string s;
  s.resize_and_overwrite(10, [](char*, std::size_t) {
    return 5.0f; // expected-error-re@*:* {{{{(static_assertion|static assertion)}}{{.*}}integer-like}}
  });
}
