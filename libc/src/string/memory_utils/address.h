//===-- Strongly typed address with alignment and access semantics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_COMMON_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_COMMON_H

#include "src/__support/CPP/TypeTraits.h"  // cpp::ConditionalType
#include "src/string/memory_utils/utils.h" // is_power2
#include <stddef.h>                        // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

namespace __llvm_libc {

// Utility to enable static_assert(false) in templates.
template <bool flag = false> static void DeferredStaticAssert(const char *msg) {
  static_assert(flag, "compilation error");
}

// A non-coercible type to represent raw data.
enum class ubyte : unsigned char { ZERO = 0 };

// Address attribute specifying whether the underlying load / store operations
// are temporal or non-temporal.
enum class Temporality { TEMPORAL, NON_TEMPORAL };

// Address attribute specifying whether the underlying load / store operations
// are aligned or unaligned.
enum class Aligned { NO, YES };

// Address attribute to discriminate between readable and writable addresses.
enum class Permission { Read, Write };

// Address is semantically equivalent to a pointer but also conveys compile time
// information that helps with instructions selection (aligned/unaligned,
// temporal/non-temporal).
template <size_t Alignment, Permission P, Temporality TS> struct Address {
  static_assert(is_power2(Alignment));
  static constexpr size_t ALIGNMENT = Alignment;
  static constexpr Permission PERMISSION = P;
  static constexpr Temporality TEMPORALITY = TS;
  static constexpr bool IS_READ = P == Permission::Read;
  static constexpr bool IS_WRITE = P == Permission::Write;
  using PointeeType = cpp::ConditionalType<!IS_WRITE, const ubyte, ubyte>;
  using VoidType = cpp::ConditionalType<!IS_WRITE, const void, void>;

  Address(VoidType *ptr) : ptr_(reinterpret_cast<PointeeType *>(ptr)) {}

  PointeeType *ptr() const {
    return reinterpret_cast<PointeeType *>(
        __builtin_assume_aligned(ptr_, ALIGNMENT));
  }

  PointeeType *const ptr_;

  template <size_t ByteOffset> auto offset(size_t byte_offset) const {
    static constexpr size_t NewAlignment = commonAlign<ByteOffset>();
    return Address<NewAlignment, PERMISSION, TEMPORALITY>(ptr_ + byte_offset);
  }

private:
  static constexpr size_t gcd(size_t A, size_t B) {
    return B == 0 ? A : gcd(B, A % B);
  }

  template <size_t ByteOffset> static constexpr size_t commonAlign() {
    constexpr size_t GCD = gcd(ByteOffset, ALIGNMENT);
    if constexpr (is_power2(GCD))
      return GCD;
    else
      return 1;
  }
};

template <typename T> struct IsAddressType : public cpp::FalseValue {};
template <size_t Alignment, Permission P, Temporality TS>
struct IsAddressType<Address<Alignment, P, TS>> : public cpp::TrueValue {};

// Reinterpret the address as a pointer to T.
// This is not UB since the underlying pointer always refers to a `char` in a
// buffer of raw data.
template <typename T, typename AddrT> static T *as(AddrT addr) {
  static_assert(IsAddressType<AddrT>::Value);
  return reinterpret_cast<T *>(addr.ptr());
}

// Offsets the address by a compile time amount, this allows propagating
// alignment whenever possible.
template <size_t ByteOffset, typename AddrT>
static auto offsetAddr(AddrT addr) {
  static_assert(IsAddressType<AddrT>::Value);
  return addr.template offset<ByteOffset>(ByteOffset);
}

// Offsets the address by a runtime amount but assuming that the resulting
// address will be Alignment aligned.
template <size_t Alignment, typename AddrT>
static auto offsetAddrAssumeAligned(AddrT addr, size_t byte_offset) {
  static_assert(IsAddressType<AddrT>::Value);
  return Address<Alignment, AddrT::PERMISSION, AddrT::TEMPORALITY>(addr.ptr_ +
                                                                   byte_offset);
}

// Offsets the address by a runtime amount that is assumed to be a multiple of
// ByteOffset. This allows to propagate the address alignment whenever possible.
template <size_t ByteOffset, typename AddrT>
static auto offsetAddrMultiplesOf(AddrT addr, ptrdiff_t byte_offset) {
  static_assert(IsAddressType<AddrT>::Value);
  return addr.template offset<ByteOffset>(byte_offset);
}

// User friendly aliases for common address types.
template <size_t Alignment>
using SrcAddr = Address<Alignment, Permission::Read, Temporality::TEMPORAL>;
template <size_t Alignment>
using DstAddr = Address<Alignment, Permission::Write, Temporality::TEMPORAL>;
template <size_t Alignment>
using NtSrcAddr =
    Address<Alignment, Permission::Read, Temporality::NON_TEMPORAL>;
template <size_t Alignment>
using NtDstAddr =
    Address<Alignment, Permission::Write, Temporality::NON_TEMPORAL>;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_COMMON_H
