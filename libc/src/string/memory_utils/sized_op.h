//===-- Sized Operations --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SizedOp struct that serves as the middle end of the
// framework. It implements sized memory operations by breaking them down into
// simpler types whose availability is described in the Backend. It also
// provides a way to load and store sized chunks of memory (necessary for the
// move operation). SizedOp are the building blocks of higher order algorithms
// like HeadTail, Align or Loop.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_SIZED_OP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_SIZED_OP_H

#include <stddef.h> // size_t

#ifndef LLVM_LIBC_USE_BUILTIN_MEMCPY_INLINE
#define LLVM_LIBC_USE_BUILTIN_MEMCPY_INLINE                                    \
  __has_builtin(__builtin_memcpy_inline)
#endif // LLVM_LIBC_USE_BUILTIN_MEMCPY_INLINE

#ifndef LLVM_LIBC_USE_BUILTIN_MEMSET_INLINE
#define LLVM_LIBC_USE_BUILTIN_MEMSET_INLINE                                    \
  __has_builtin(__builtin_memset_inline)
#endif // LLVM_LIBC_USE_BUILTIN_MEMSET_INLINE

namespace __llvm_libc {

template <typename Backend, size_t Size> struct SizedOp {
  static constexpr size_t SIZE = Size;

private:
  static_assert(Backend::IS_BACKEND_TYPE);
  static_assert(SIZE > 0);
  using type = typename Backend::template getNextType<Size>;
  static constexpr size_t TYPE_SIZE = sizeof(type);
  static_assert(SIZE >= TYPE_SIZE);
  static constexpr size_t NEXT_SIZE = Size - TYPE_SIZE;
  using NextBlock = SizedOp<Backend, NEXT_SIZE>;

  // Returns whether we can use an aligned operations.
  // This is possible because the address type carries known compile-time
  // alignment informations.
  template <typename T, typename AddrT> static constexpr Aligned isAligned() {
    static_assert(IsAddressType<AddrT>::Value);
    return AddrT::ALIGNMENT > 1 && AddrT::ALIGNMENT >= sizeof(T) ? Aligned::YES
                                                                 : Aligned::NO;
  }

  // Loads a value of the current `type` from `src`.
  // This function is responsible for extracting Temporality and Alignment from
  // the Address type.
  template <typename SrcAddrT> static inline auto nativeLoad(SrcAddrT src) {
    static_assert(IsAddressType<SrcAddrT>::Value && SrcAddrT::IS_READ);
    constexpr auto AS = isAligned<type, SrcAddrT>();
    constexpr auto TS = SrcAddrT::TEMPORALITY;
    return Backend::template load<type, TS, AS>(as<const type>(src));
  }

  // Stores a value of the current `type` to `dst`.
  // This function is responsible for extracting Temporality and Alignment from
  // the Address type.
  template <typename DstAddrT>
  static inline void nativeStore(type value, DstAddrT dst) {
    static_assert(IsAddressType<DstAddrT>::Value && DstAddrT::IS_WRITE);
    constexpr auto AS = isAligned<type, DstAddrT>();
    constexpr auto TS = DstAddrT::TEMPORALITY;
    return Backend::template store<type, TS, AS>(as<type>(dst), value);
  }

  // A well aligned POD structure to store Size bytes.
  // This is used to implement the move operations.
  struct Value {
    alignas(alignof(type)) ubyte payload[Size];
  };

public:
  template <typename DstAddrT, typename SrcAddrT>
  static inline void copy(DstAddrT dst, SrcAddrT src) {
    static_assert(IsAddressType<DstAddrT>::Value && DstAddrT::IS_WRITE);
    static_assert(IsAddressType<SrcAddrT>::Value && SrcAddrT::IS_READ);
    if constexpr (LLVM_LIBC_USE_BUILTIN_MEMCPY_INLINE &&
                  DstAddrT::TEMPORALITY == Temporality::TEMPORAL &&
                  SrcAddrT::TEMPORALITY == Temporality::TEMPORAL) {
      // delegate optimized copy to compiler.
      __builtin_memcpy_inline(dst.ptr(), src.ptr(), Size);
      return;
    }
    nativeStore(nativeLoad(src), dst);
    if constexpr (NEXT_SIZE > 0)
      NextBlock::copy(offsetAddr<TYPE_SIZE>(dst), offsetAddr<TYPE_SIZE>(src));
  }

  template <typename DstAddrT, typename SrcAddrT>
  static inline void move(DstAddrT dst, SrcAddrT src) {
    const auto payload = nativeLoad(src);
    if constexpr (NEXT_SIZE > 0)
      NextBlock::move(offsetAddr<TYPE_SIZE>(dst), offsetAddr<TYPE_SIZE>(src));
    nativeStore(payload, dst);
  }

  template <typename DstAddrT>
  static inline void set(DstAddrT dst, ubyte value) {
    if constexpr (LLVM_LIBC_USE_BUILTIN_MEMSET_INLINE &&
                  DstAddrT::TEMPORALITY == Temporality::TEMPORAL) {
      // delegate optimized set to compiler.
      __builtin_memset_inline(dst.ptr(), value, Size);
      return;
    }
    nativeStore(Backend::template splat<type>(value), dst);
    if constexpr (NEXT_SIZE > 0)
      NextBlock::set(offsetAddr<TYPE_SIZE>(dst), value);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2) {
    const uint64_t current =
        Backend::template notEquals<type>(nativeLoad(src1), nativeLoad(src2));
    if constexpr (NEXT_SIZE > 0) {
      // In the case where we cannot handle Size with single operation (e.g.
      // Size == 3) we can either return early if current is non zero or
      // aggregate all the operations through the bitwise or operator.
      // We chose the later to reduce branching.
      return current | (NextBlock::isDifferent(offsetAddr<TYPE_SIZE>(src1),
                                               offsetAddr<TYPE_SIZE>(src2)));
    } else {
      return current;
    }
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2) {
    const auto a = nativeLoad(src1);
    const auto b = nativeLoad(src2);
    // If we cannot handle Size as a single operation we have two choices:
    // - Either use Backend's threeWayCmp directly and return it is non
    // zero.
    //
    //   if (int32_t res = Backend::template threeWayCmp<type>(a, b))
    //     return res;
    //
    // - Or use Backend's notEquals first and use threeWayCmp only if
    // different, the assumption here is that notEquals is faster than
    // threeWayCmp and that we can save cycles when the Size needs to be
    // decomposed in many sizes (e.g. Size == 7 => 4 + 2 + 1)
    //
    //   if (Backend::template notEquals<type>(a, b))
    //     return Backend::template threeWayCmp<type>(a, b);
    //
    // We chose the former to reduce code bloat and branching.
    if (int32_t res = Backend::template threeWayCmp<type>(a, b))
      return res;
    if constexpr (NEXT_SIZE > 0)
      return NextBlock::threeWayCmp(offsetAddr<TYPE_SIZE>(src1),
                                    offsetAddr<TYPE_SIZE>(src2));
    return 0;
  }

  template <typename SrcAddrT> static Value load(SrcAddrT src) {
    Value output;
    copy(DstAddr<alignof(type)>(output.payload), src);
    return output;
  }

  template <typename DstAddrT> static void store(DstAddrT dst, Value value) {
    copy(dst, SrcAddr<alignof(type)>(value.payload));
  }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_SIZED_OP_H
