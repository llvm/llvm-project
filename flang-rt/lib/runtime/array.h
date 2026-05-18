//===-- lib/runtime/array.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_ARRAY_H_
#define FLANG_RT_RUNTIME_ARRAY_H_

#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"

namespace Fortran::runtime {
// A simple dynamic array that only supports appending to avoid std::vector.
template <typename T> struct DynamicArray {
  ~DynamicArray() {
    for (std::size_t i = 0; i < size_; ++i) {
      data_[i].~T();
    }
    FreeMemory(data_);
  }

  void emplace_back(T &&value) {
    if (size_ == capacity_) {
      reserve(capacity_ ? capacity_ * 2 : 4);
    }
    new (data_ + size_) T(std::move(value));
    ++size_;
  }

  void reserve(std::size_t newCap) {
    if (newCap <= capacity_) {
      return;
    }
    T *new_data = static_cast<T *>(
        AllocateMemoryOrCrash(terminator_, newCap * sizeof(T)));
    for (std::size_t i = 0; i < size_; ++i) {
      new (new_data + i) T(std::move(data_[i]));
      data_[i].~T();
    }
    FreeMemory(data_);
    data_ = new_data;
    capacity_ = newCap;
  }

  T *begin() const { return data_; }
  T *end() const { return data_ + size_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
  Terminator terminator_{__FILE__, __LINE__};
};
} // namespace Fortran::runtime

#endif // FLANG_RT_RUNTIME_ARRAY_H_
