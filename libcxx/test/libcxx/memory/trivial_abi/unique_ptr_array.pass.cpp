//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test unique_ptr<T[]> with trivial_abi as parameter type.

// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI

// XFAIL: gcc
// UNSUPPORTED: c++03

#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Node {
  int* shared_val;

  explicit Node(int* ptr) : shared_val(ptr) {}
  ~Node() { ++(*shared_val); }
};

__attribute__((noinline)) bool get_val(std::unique_ptr<Node[]> /*unused*/) {
  call_something();
  return true;
}

__attribute__((noinline)) void expect_3(int* shared, bool /*unused*/) {
  assert(*shared == 3);
}

int main(int, char**) {
  int shared = 0;

  // Without trivial-abi, the unique_ptr is deleted at the end of this
  // statement, expect_3 will see shared == 0 because it's not incremented (in
  // ~Node()) until the end of this statement.
  //
  // With trivial-abi, shared_val is incremented 3 times before get_val returns
  // because ~Node() was called 3 times.
  expect_3(&shared, get_val(std::unique_ptr<Node[]>(new Node[3]{
                        Node(&shared), Node(&shared), Node(&shared)})));

  // Check that shared_value is still 3 (ie., ~Node() isn't called again by the end of the full-expression above)
  expect_3(&shared, true);

  return 0;
}
