//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test unique_ptr<T> with trivial_abi as parameter type.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI

// There were assertion failures in both parse and codegen, which are fixed in clang 11.
// UNSUPPORTED: gcc, clang-4, clang-5, clang-6, clang-7, clang-8, clang-9, clang-10

#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Node {
  int* shared_val;

  explicit Node(int* ptr) : shared_val(ptr) {}
  ~Node() { ++(*shared_val); }
};

__attribute__((noinline)) bool get_val(std::unique_ptr<Node> /*unused*/) {
  call_something();
  return true;
}

__attribute__((noinline)) void expect_1(int* shared, bool /*unused*/) {
  assert(*shared == 1);
}

int main(int, char**) {
  int shared = 0;

  // Without trivial-abi, the unique_ptr is deleted at the end of this
  // statement; expect_1 will see shared == 0 because it's not incremented (in
  // ~Node()) until expect_1 returns.
  //
  // With trivial-abi, expect_1 will see shared == 1 because shared_val is
  // incremented before get_val returns.
  expect_1(&shared, get_val(std::unique_ptr<Node>(new Node(&shared))));

  // Check that the shared-value is still 1.
  expect_1(&shared, true);
  return 0;
}
