//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
//
// REQUIRES: c++03 || c++11 || c++14
//
// <future>
//
// Ensure that packaged_task destroys its members in the correct order.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <future>

#include "test_macros.h"

template <int Align, int BufferSize>
struct TEST_ALIGNAS(Align) CallableWithPadding {
  CallableWithPadding() {} // Allow putting object into the tail padding
  CallableWithPadding(const CallableWithPadding&) {}
  CallableWithPadding& operator=(const CallableWithPadding&) { return *this; }
  ~CallableWithPadding() { std::memset(static_cast<void*>(this), 0, sizeof(*this)); }

  void operator()() {}

  char buffer[BufferSize];
};

template <class T>
struct AllocatorWithData {
  using value_type = T;

  AllocatorWithData() : data(123) {}
  AllocatorWithData(const AllocatorWithData& other) : data(other.data) {}
  AllocatorWithData& operator=(const AllocatorWithData& other) { data = other.data; }
  ~AllocatorWithData() { assert(data == 123); }

  template <class U>
  AllocatorWithData(const AllocatorWithData<U>& other) : data(other.data) {}

  uint64_t data;

  T* allocate(size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* ptr, size_t n) { std::allocator<T>().deallocate(ptr, n); }
};

int main(int, char**) {
  std::packaged_task<void()> heap((std::allocator_arg_t(), AllocatorWithData<int>(), CallableWithPadding<32, 24>()));
  std::packaged_task<void()> stack((std::allocator_arg_t(), AllocatorWithData<int>(), CallableWithPadding<16, 8>()));
  return 0;
}
