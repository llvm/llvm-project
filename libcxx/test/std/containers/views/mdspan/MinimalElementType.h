//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H

#include<memory>

// Idiosyncratic element type for mdspan
// Make sure we don't assume copyable, default constructible, movable etc.
struct MinimalElementType {
  int val;
  constexpr MinimalElementType() = delete;
  constexpr MinimalElementType(const MinimalElementType&) = delete;
  constexpr explicit MinimalElementType(int v) noexcept : val(v){}
  constexpr MinimalElementType& operator=(const MinimalElementType&) = delete;
};

// Helper class to create pointer to MinimalElementType
template<class T, size_t N>
struct ElementPool {
  constexpr ElementPool() {
    ptr_ = std::allocator<T>().allocate(N);
    for (int i = 0; i != N; ++i)
      std::construct_at(ptr_ + i, 42);
  }

  constexpr T* get_ptr() { return ptr_; }

  constexpr ~ElementPool() {
    for (int i = 0; i != N; ++i)
      std::destroy_at(ptr_ + i);
    std::allocator<T>().deallocate(ptr_, N);
  }

private:
  T* ptr_;
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
