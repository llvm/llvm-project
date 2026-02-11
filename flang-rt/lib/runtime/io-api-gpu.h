//===-- lib/runtime/io-api-gpu.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_IO_API_GPU_H_
#define FLANG_RT_RUNTIME_IO_API_GPU_H_

#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include <cstdint>
#include <utility>

namespace Fortran::runtime::io {
// We reserve the RPC opcodes with 'f' in the MSB for Fortran usage.
constexpr std::uint32_t MakeOpcode(std::uint32_t base) {
  return ('f' << 24) | base;
}

// Opcodes shared between the client and server for each function we support.
constexpr std::uint32_t BeginExternalListOutput_Opcode = MakeOpcode(0);
constexpr std::uint32_t EndIoStatement_Opcode = MakeOpcode(1);
constexpr std::uint32_t OutputInteger8_Opcode = MakeOpcode(2);
constexpr std::uint32_t OutputInteger16_Opcode = MakeOpcode(3);
constexpr std::uint32_t OutputInteger32_Opcode = MakeOpcode(4);
constexpr std::uint32_t OutputInteger64_Opcode = MakeOpcode(5);
constexpr std::uint32_t OutputInteger128_Opcode = MakeOpcode(6);
constexpr std::uint32_t OutputReal32_Opcode = MakeOpcode(7);
constexpr std::uint32_t OutputReal64_Opcode = MakeOpcode(8);
constexpr std::uint32_t OutputComplex32_Opcode = MakeOpcode(9);
constexpr std::uint32_t OutputComplex64_Opcode = MakeOpcode(10);
constexpr std::uint32_t OutputAscii_Opcode = MakeOpcode(11);
constexpr std::uint32_t OutputLogical_Opcode = MakeOpcode(12);

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

} // namespace Fortran::runtime::io

#endif // FLANG_RT_RUNTIME_IO_API_GPU_H_
