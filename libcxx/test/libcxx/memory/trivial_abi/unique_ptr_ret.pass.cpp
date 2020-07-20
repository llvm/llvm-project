//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test unique_ptr<T> with trivial_abi as return-type.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI

// There were assertion failures in both parse and codegen, which are fixed in clang 11.
// UNSUPPORTED: gcc, clang-4, clang-5, clang-6, clang-7, clang-8, clang-9, clang-10

#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Node {
  explicit Node() {}
  ~Node() {}
};

__attribute__((noinline)) std::unique_ptr<Node> make_val(void** local_addr) {
  call_something();

  auto ret = std::make_unique<Node>();

  // Capture the local address of ret.
  *local_addr = &ret;

  return ret;
}

int main(int, char**) {
  void* local_addr = nullptr;
  auto ret = make_val(&local_addr);
  assert(local_addr != nullptr);

  // Without trivial_abi, &ret == local_addr because the return value
  // is allocated here in main's stackframe.
  //
  // With trivial_abi, local_addr is the address of a local variable in
  // make_val, and hence different from &ret.
  assert((void*)&ret != local_addr);

  return 0;
}
