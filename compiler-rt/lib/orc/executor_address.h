//===------ ExecutorAddress.h - Executing process address -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents an address in the executing program.
//
// This file was derived from
// llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_EXECUTOR_ADDRESS_H
#define ORC_RT_EXECUTOR_ADDRESS_H

#include "adt.h"
#include "simple_packed_serialization.h"

#include <cassert>
#include <type_traits>

namespace __orc_rt {

/// Represents the difference between two addresses in the executor process.
class ExecutorAddrDiff {
public:
  ExecutorAddrDiff() = default;
  explicit ExecutorAddrDiff(uint64_t Value) : Value(Value) {}

  uint64_t getValue() const { return Value; }

private:
  int64_t Value = 0;
};

/// Represents an address in the executor process.
class ExecutorAddress {
public:
  ExecutorAddress() = default;
  explicit ExecutorAddress(uint64_t Addr) : Addr(Addr) {}

  /// Create an ExecutorAddress from the given pointer.
  /// Warning: This should only be used when JITing in-process.
  template <typename T> static ExecutorAddress fromPtr(T *Value) {
    return ExecutorAddress(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Value)));
  }

  /// Cast this ExecutorAddress to a pointer of the given type.
  /// Warning: This should only be esude when JITing in-process.
  template <typename T> T toPtr() const {
    static_assert(std::is_pointer<T>::value, "T must be a pointer type");
    uintptr_t IntPtr = static_cast<uintptr_t>(Addr);
    assert(IntPtr == Addr &&
           "JITTargetAddress value out of range for uintptr_t");
    return reinterpret_cast<T>(IntPtr);
  }

  uint64_t getValue() const { return Addr; }
  void setValue(uint64_t Addr) { this->Addr = Addr; }
  bool isNull() const { return Addr == 0; }

  explicit operator bool() const { return Addr != 0; }

  friend bool operator==(const ExecutorAddress &LHS,
                         const ExecutorAddress &RHS) {
    return LHS.Addr == RHS.Addr;
  }

  friend bool operator!=(const ExecutorAddress &LHS,
                         const ExecutorAddress &RHS) {
    return LHS.Addr != RHS.Addr;
  }

  friend bool operator<(const ExecutorAddress &LHS,
                        const ExecutorAddress &RHS) {
    return LHS.Addr < RHS.Addr;
  }

  friend bool operator<=(const ExecutorAddress &LHS,
                         const ExecutorAddress &RHS) {
    return LHS.Addr <= RHS.Addr;
  }

  friend bool operator>(const ExecutorAddress &LHS,
                        const ExecutorAddress &RHS) {
    return LHS.Addr > RHS.Addr;
  }

  friend bool operator>=(const ExecutorAddress &LHS,
                         const ExecutorAddress &RHS) {
    return LHS.Addr >= RHS.Addr;
  }

  ExecutorAddress &operator++() {
    ++Addr;
    return *this;
  }
  ExecutorAddress &operator--() {
    --Addr;
    return *this;
  }
  ExecutorAddress operator++(int) { return ExecutorAddress(Addr++); }
  ExecutorAddress operator--(int) { return ExecutorAddress(Addr++); }

  ExecutorAddress &operator+=(const ExecutorAddrDiff Delta) {
    Addr += Delta.getValue();
    return *this;
  }

  ExecutorAddress &operator-=(const ExecutorAddrDiff Delta) {
    Addr -= Delta.getValue();
    return *this;
  }

private:
  uint64_t Addr = 0;
};

/// Subtracting two addresses yields an offset.
inline ExecutorAddrDiff operator-(const ExecutorAddress &LHS,
                                  const ExecutorAddress &RHS) {
  return ExecutorAddrDiff(LHS.getValue() - RHS.getValue());
}

/// Adding an offset and an address yields an address.
inline ExecutorAddress operator+(const ExecutorAddress &LHS,
                                 const ExecutorAddrDiff &RHS) {
  return ExecutorAddress(LHS.getValue() + RHS.getValue());
}

/// Adding an address and an offset yields an address.
inline ExecutorAddress operator+(const ExecutorAddrDiff &LHS,
                                 const ExecutorAddress &RHS) {
  return ExecutorAddress(LHS.getValue() + RHS.getValue());
}

/// Represents an address range in the exceutor process.
struct ExecutorAddressRange {
  ExecutorAddressRange() = default;
  ExecutorAddressRange(ExecutorAddress StartAddress, ExecutorAddress EndAddress)
      : StartAddress(StartAddress), EndAddress(EndAddress) {}

  bool empty() const { return StartAddress == EndAddress; }
  ExecutorAddrDiff size() const { return EndAddress - StartAddress; }

  template <typename T> span<T> toSpan() const {
    assert(size().getValue() % sizeof(T) == 0 &&
           "AddressRange is not a multiple of sizeof(T)");
    return span<T>(StartAddress.toPtr<T *>(), size().getValue() / sizeof(T));
  }

  ExecutorAddress StartAddress;
  ExecutorAddress EndAddress;
};

/// SPS serializatior for ExecutorAddress.
template <> class SPSSerializationTraits<SPSExecutorAddress, ExecutorAddress> {
public:
  static size_t size(const ExecutorAddress &EA) {
    return SPSArgList<uint64_t>::size(EA.getValue());
  }

  static bool serialize(SPSOutputBuffer &BOB, const ExecutorAddress &EA) {
    return SPSArgList<uint64_t>::serialize(BOB, EA.getValue());
  }

  static bool deserialize(SPSInputBuffer &BIB, ExecutorAddress &EA) {
    uint64_t Tmp;
    if (!SPSArgList<uint64_t>::deserialize(BIB, Tmp))
      return false;
    EA = ExecutorAddress(Tmp);
    return true;
  }
};

using SPSExecutorAddressRange =
    SPSTuple<SPSExecutorAddress, SPSExecutorAddress>;

/// Serialization traits for address ranges.
template <>
class SPSSerializationTraits<SPSExecutorAddressRange, ExecutorAddressRange> {
public:
  static size_t size(const ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::size(
        Value.StartAddress, Value.EndAddress);
  }

  static bool serialize(SPSOutputBuffer &BOB,
                        const ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::serialize(
        BOB, Value.StartAddress, Value.EndAddress);
  }

  static bool deserialize(SPSInputBuffer &BIB, ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::deserialize(
        BIB, Value.StartAddress, Value.EndAddress);
  }
};

using SPSExecutorAddressRangeSequence = SPSSequence<SPSExecutorAddressRange>;

} // End namespace __orc_rt

#endif // ORC_RT_EXECUTOR_ADDRESS_H
