//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test arguments destruction order involving unique_ptr<T> with trivial_abi.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI

// There were assertion failures in both parse and codegen, which are fixed in clang 11.
// UNSUPPORTED: gcc, clang-4, clang-5, clang-6, clang-7, clang-8, clang-9, clang-10


#include <memory>
#include <cassert>

__attribute__((noinline)) void call_something() { asm volatile(""); }

struct Base {
  char* shared_buff;
  int* cur_idx;
  const char id;

  explicit Base(char* buf, int* idx, char ch)
      : shared_buff(buf), cur_idx(idx), id(ch) {}
  ~Base() { shared_buff[(*cur_idx)++] = id; }
};

struct A : Base {
  explicit A(char* buf, int* idx) : Base(buf, idx, 'A') {}
};

struct B : Base {
  explicit B(char* buf, int* idx) : Base(buf, idx, 'B') {}
};

struct C : Base {
  explicit C(char* buf, int* idx) : Base(buf, idx, 'C') {}
};

__attribute__((noinline)) void func(A /*unused*/, std::unique_ptr<B> /*unused*/,
                                    C /*unused*/) {
  call_something();
}

int main(int, char**) {
  char shared_buf[3] = {'0', '0', '0'};
  int cur_idx = 0;

  func(A(shared_buf, &cur_idx), std::unique_ptr<B>(new B(shared_buf, &cur_idx)),
       C(shared_buf, &cur_idx));

#if defined(_LIBCPP_ABI_MICROSOFT)
  // On Microsoft ABI, the dtor order is always A,B,C (because callee-destroyed)
  assert(shared_buf[0] == 'A' && shared_buf[1] == 'B' && shared_buf[2] == 'C');
#else
  // With trivial_abi, the std::unique_ptr<B> arg is always destructed first.
  assert(shared_buf[0] == 'B');
#endif
  return 0;
}
