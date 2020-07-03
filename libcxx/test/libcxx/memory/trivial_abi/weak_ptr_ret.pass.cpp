//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test weak_ptr<T> with trivial_abi as return-type.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_ENABLE_SHARED_PTR_TRIVIAL_ABI

#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Node {
  explicit Node() {}
  ~Node() {}
};

__attribute__((noinline)) std::weak_ptr<Node>
make_val(std::shared_ptr<Node>& sptr, void** local_addr) {
  call_something();

  std::weak_ptr<Node> ret;
  ret = sptr;

  // Capture the local address of ret.
  *local_addr = &ret;

  return ret;
}

int main(int, char**) {
  void* local_addr = nullptr;
  auto sptr = std::make_shared<Node>(&shared);
  std::weak_ptr<Node> ret = make_val(sptr, &local_addr);
  assert(local_addr != nullptr);

  // Without trivial_abi, &ret == local_addr because the return value
  // is allocated here in main's stackframe.
  //
  // With trivial_abi, local_addr is the address of a local variable in
  // make_val, and hence different from &ret.
  assert((void*)&ret != local_addr);

  return 0;
}
