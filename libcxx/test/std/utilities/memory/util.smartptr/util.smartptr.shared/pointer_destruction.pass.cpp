//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <memory>
//
// class shared_ptr
//
// Ensure that any constructed (fancy) pointer objects are destroyed.

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <memory>
#include <utility>

#include "test_macros.h"

static size_t pointer_count = 0;

template <class T>
struct Pointer {
  T* ptr_ = nullptr;

  Pointer() : ptr_(nullptr) { ++pointer_count; }
  Pointer(T* ptr) : ptr_(ptr) { ++pointer_count; }
  Pointer(nullptr_t) : ptr_(nullptr) { ++pointer_count; }
  Pointer(const Pointer& other) : ptr_(other.ptr_) { ++pointer_count; }
  Pointer& operator=(const Pointer& other) {
    ptr_ = other.ptr_;
    return *this;
  }
  ~Pointer() { --pointer_count; }

  Pointer& operator=(nullptr_t) {
    ptr_ = nullptr;
    return *this;
  }

  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }

  explicit operator bool() const { return ptr_; }

  operator T*() { return ptr_; }

  friend bool operator==(const Pointer& lhs, const Pointer& rhs) { return lhs.ptr_ == rhs.ptr_; }
  friend bool operator!=(const Pointer& lhs, const Pointer& rhs) { return !(lhs == rhs); }
  friend bool operator==(const Pointer& lhs, nullptr_t) { return lhs == Pointer(nullptr); }
  friend bool operator!=(const Pointer& lhs, nullptr_t) { return !(lhs == nullptr); }

  static Pointer pointer_to(T& ref) { return Pointer(std::addressof(ref)); }
};

template <class T>
struct Deleter {
  using pointer = Pointer<T>;

  void operator()(pointer ptr) const { delete ptr.ptr_; }
};

int main(int, char**) {
  {
    auto ptr = std::unique_ptr<int, Deleter<int> >(new int());
    auto v   = std::shared_ptr<int>(std::move(ptr));
    LIBCPP_ASSERT(pointer_count != 0);
  }
  assert(pointer_count == 0);

  return 0;
}
