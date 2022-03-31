//===-- A self contained equivalent of std::vector --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_VECTOR_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_VECTOR_H

#include <stddef.h> // For size_t.

#include <stdlib.h> // For malloc/realloc/free

namespace __llvm_libc {
namespace cpp {

// This implementation does not have a templated allocator since that feature
// isn't relevant for a libc setting.

// Vector is a templated dynamically resizable array. This implementation is
// only meant for primitives or structs, and will not call destructors on held
// objects.
template <class T> class vector {
  T *data_array;
  size_t array_size;
  size_t num_elements = 0;
  static constexpr size_t DEFAULT_SIZE = 16;
  static constexpr size_t GROWTH_FACTOR = 2;
  static constexpr size_t MAX_SIZE = ~size_t(0);

public:
  constexpr vector<T>() : array_size{DEFAULT_SIZE} {
    data_array = static_cast<T *>(malloc(DEFAULT_SIZE * sizeof(T)));
  }

  constexpr vector<T>(const vector<T> &other) = delete;
  constexpr vector<T>(const vector<T> &&other) = delete;

  ~vector() { free(data_array); }

  constexpr vector &operator=(vector &other) = delete;
  constexpr vector &operator=(vector &&other) = delete;

  constexpr void reserve(size_t new_size) {
    if (new_size >= array_size)
      increase_size(new_size + 1);
  }

  constexpr void push_back(const T &value) {
    if (num_elements >= array_size)
      increase_size(num_elements + 1);
    data_array[num_elements] = value;
    ++num_elements;
  }

  constexpr T &operator[](size_t pos) { return data_array[pos]; }
  constexpr T *data() { return data_array; }
  constexpr const T *data() const { return data_array; }

  constexpr bool empty() const { return num_elements == 0; }

  constexpr size_t size() const { return num_elements; }
  constexpr size_t max_size() const { return MAX_SIZE; }

  constexpr size_t capacity() const { return array_size; }

private:
  static constexpr size_t MAX_DIV_BY_GROWTH = MAX_SIZE / GROWTH_FACTOR;

  // new_size is treated as the minimum size for the new array. This function
  // will increase array_size by GROWTH_FACTOR until there is space for new_size
  // items.
  constexpr void increase_size(size_t new_size) {
    size_t temp_size = array_size;
    if (new_size >= MAX_DIV_BY_GROWTH) {
      temp_size = new_size;
    } else {
      if (temp_size == 0)
        temp_size = 1;
      while (temp_size <= new_size)
        temp_size = temp_size * GROWTH_FACTOR;
    }
    array_size = temp_size;
    data_array = static_cast<T *>(realloc(data_array, array_size * sizeof(T)));
  }
};
} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_VECTOR_H
