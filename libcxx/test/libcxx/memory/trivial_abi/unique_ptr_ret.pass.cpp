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

// XFAIL: gcc

#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Node {
  explicit Node() {}
  ~Node() {}
};

__attribute__((noinline)) std::unique_ptr<Node> make_val(void** local_addr) {
  call_something();

  auto ret = std::unique_ptr<Node>(new Node);

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
#if !defined(__i386__) && !defined(_WIN32) && !defined(_AIX)
  // On X86, structs are never returned in registers.
  // On AIX, structs are never returned in registers.
  // Thus, unique_ptr will be passed indirectly even if it is trivial.
  // On Windows, structs with a destructor are always returned indirectly.
  assert((void*)&ret != local_addr);
#endif

  return 0;
}
