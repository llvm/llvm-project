//===-- Generic implementation of memory function building blocks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides generic C++ building blocks.
// Depending on the requested size, the block operation uses unsigned integral
// types, vector types or an array of the type with the maximum size.
//
// The maximum size is passed as a template argument. For instance, on x86
// platforms that only supports integral types the maximum size would be 8
// (corresponding to uint64_t). On this platform if we request the size 32, this
// would be treated as a cpp::array<uint64_t, 4>.
//
// On the other hand, if the platform is x86 with support for AVX the maximum
// size is 32 and the operation can be handled with a single native operation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_GENERIC_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_GENERIC_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/endian.h"
#include "src/__support/macros/optimization.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/utils.h"

#include <stdint.h>

namespace __llvm_libc::generic {

// Implements generic load, store, splat and set for unsigned integral types.
template <typename T> struct ScalarType {
  static_assert(cpp::is_integral_v<T> && !cpp::is_signed_v<T>);
  using Type = T;
  static constexpr size_t SIZE = sizeof(Type);

  LIBC_INLINE static Type load(CPtr src) {
    return ::__llvm_libc::load<Type>(src);
  }
  LIBC_INLINE static void store(Ptr dst, Type value) {
    ::__llvm_libc::store<Type>(dst, value);
  }
  LIBC_INLINE static Type splat(uint8_t value) {
    return Type(~0) / Type(0xFF) * Type(value);
  }
  LIBC_INLINE static void set(Ptr dst, uint8_t value) {
    store(dst, splat(value));
  }
};

// GCC can only take literals as __vector_size__ argument so we have to use
// template specialization.
template <size_t Size> struct VectorValueType {};
template <> struct VectorValueType<1> {
  using type = uint8_t __attribute__((__vector_size__(1)));
};
template <> struct VectorValueType<2> {
  using type = uint8_t __attribute__((__vector_size__(2)));
};
template <> struct VectorValueType<4> {
  using type = uint8_t __attribute__((__vector_size__(4)));
};
template <> struct VectorValueType<8> {
  using type = uint8_t __attribute__((__vector_size__(8)));
};
template <> struct VectorValueType<16> {
  using type = uint8_t __attribute__((__vector_size__(16)));
};
template <> struct VectorValueType<32> {
  using type = uint8_t __attribute__((__vector_size__(32)));
};
template <> struct VectorValueType<64> {
  using type = uint8_t __attribute__((__vector_size__(64)));
};

// Implements generic load, store, splat and set for vector types.
template <size_t Size> struct VectorType {
  using Type = typename VectorValueType<Size>::type;
  static constexpr size_t SIZE = Size;
  LIBC_INLINE static Type load(CPtr src) {
    return ::__llvm_libc::load<Type>(src);
  }
  LIBC_INLINE static void store(Ptr dst, Type value) {
    ::__llvm_libc::store<Type>(dst, value);
  }
  LIBC_INLINE static Type splat(uint8_t value) {
    Type Out;
    // This for loop is optimized out for vector types.
    for (size_t i = 0; i < Size; ++i)
      Out[i] = static_cast<uint8_t>(value);
    return Out;
  }
  LIBC_INLINE static void set(Ptr dst, uint8_t value) {
    store(dst, splat(value));
  }
};

// Implements load, store and set for sizes not natively supported by the
// platform. SubType is either ScalarType or VectorType.
template <typename SubType, size_t ArraySize> struct ArrayType {
  using Type = cpp::array<typename SubType::Type, ArraySize>;
  static constexpr size_t SizeOfElement = SubType::SIZE;
  static constexpr size_t SIZE = SizeOfElement * ArraySize;
  LIBC_INLINE static Type load(CPtr src) {
    Type Value;
    for (size_t I = 0; I < ArraySize; ++I)
      Value[I] = SubType::load(src + (I * SizeOfElement));
    return Value;
  }
  LIBC_INLINE static void store(Ptr dst, Type Value) {
    for (size_t I = 0; I < ArraySize; ++I)
      SubType::store(dst + (I * SizeOfElement), Value[I]);
  }
  LIBC_INLINE static void set(Ptr dst, uint8_t value) {
    const auto Splat = SubType::splat(value);
    for (size_t I = 0; I < ArraySize; ++I)
      SubType::store(dst + (I * SizeOfElement), Splat);
  }
};

static_assert((UINTPTR_MAX == 4294967295U) ||
                  (UINTPTR_MAX == 18446744073709551615UL),
              "We currently only support 32- or 64-bit platforms");

#if defined(LIBC_TARGET_ARCH_IS_X86_64) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
#define LLVM_LIBC_HAS_UINT64
#endif

namespace details {
// Checks that each type's SIZE is sorted in strictly decreasing order.
// i.e. First::SIZE > Second::SIZE > ... > Last::SIZE
template <typename First, typename Second, typename... Next>
constexpr bool is_decreasing_size() {
  if constexpr (sizeof...(Next) > 0) {
    return (First::SIZE > Second::SIZE) &&
           is_decreasing_size<Second, Next...>();
  } else {
    return First::SIZE > Second::SIZE;
  }
}

// Helper to test if a type is void.
template <typename T> inline constexpr bool is_void_v = cpp::is_same_v<T, void>;

} // namespace details

// 'SupportedTypes' holds a list of natively supported types.
// The types are instanciations of ScalarType or VectorType.
// They should be ordered in strictly decreasing order.
// The 'TypeFor<Size>' type retrieves is the largest supported type that can
// handle 'Size' bytes. e.g.
//
// using ST = SupportedTypes<ScalarType<uint16_t>, ScalarType<uint8_t>>;
// using Type = ST::TypeFor<10>;
// static_assert(cpp:is_same_v<Type, ScalarType<uint16_t>>);
template <typename First, typename Second, typename... Next>
struct SupportedTypes {
  static_assert(details::is_decreasing_size<First, Second, Next...>());
  using MaxType = First;

  template <size_t Size>
  using TypeFor = cpp::conditional_t<
      (Size >= First::SIZE), First,
      typename SupportedTypes<Second, Next...>::template TypeFor<Size>>;
};

template <typename First, typename Second>
struct SupportedTypes<First, Second> {
  static_assert(details::is_decreasing_size<First, Second>());
  using MaxType = First;

  template <size_t Size>
  using TypeFor = cpp::conditional_t<
      (Size >= First::SIZE), First,
      cpp::conditional_t<(Size >= Second::SIZE), Second, void>>;
};

// Map from sizes to structures offering static load, store and splat methods.
// Note: On platforms lacking vector support, we use the ArrayType below and
// decompose the operation in smaller pieces.

// Lists a generic native types to use for Memset and Memmove operations.
// TODO: Inject the native types within Memset and Memmove depending on the
// target architectures and derive MaxSize from it.
using NativeTypeMap =
    SupportedTypes<VectorType<64>, //
                   VectorType<32>, //
                   VectorType<16>,
#if defined(LLVM_LIBC_HAS_UINT64)
                   ScalarType<uint64_t>, // Not available on 32bit
#endif
                   ScalarType<uint32_t>, //
                   ScalarType<uint16_t>, //
                   ScalarType<uint8_t>>;

namespace details {

// In case the 'Size' is not supported we can fall back to a sequence of smaller
// operations using the largest natively supported type.
template <size_t Size, size_t MaxSize> static constexpr bool useArrayType() {
  return (Size > MaxSize) && ((Size % MaxSize) == 0) &&
         !details::is_void_v<NativeTypeMap::TypeFor<MaxSize>>;
}

// Compute the type to handle an operation of 'Size' bytes knowing that the
// underlying platform only support native types up to MaxSize bytes.
template <size_t Size, size_t MaxSize>
using getTypeFor = cpp::conditional_t<
    useArrayType<Size, MaxSize>(),
    ArrayType<NativeTypeMap::TypeFor<MaxSize>, Size / MaxSize>,
    NativeTypeMap::TypeFor<Size>>;

} // namespace details


///////////////////////////////////////////////////////////////////////////////
// Memset
// The MaxSize template argument gives the maximum size handled natively by the
// platform. For instance on x86 with AVX support this would be 32. If a size
// greater than MaxSize is requested we break the operation down in smaller
// pieces of size MaxSize.
///////////////////////////////////////////////////////////////////////////////
template <size_t Size, size_t MaxSize> struct Memset {
  static_assert(is_power2(MaxSize));
  static constexpr size_t SIZE = Size;

  LIBC_INLINE static void block(Ptr dst, uint8_t value) {
    if constexpr (Size == 3) {
      Memset<1, MaxSize>::block(dst + 2, value);
      Memset<2, MaxSize>::block(dst, value);
    } else {
      using T = details::getTypeFor<Size, MaxSize>;
      if constexpr (details::is_void_v<T>) {
        deferred_static_assert("Unimplemented Size");
      } else {
        T::set(dst, value);
      }
    }
  }

  LIBC_INLINE static void tail(Ptr dst, uint8_t value, size_t count) {
    block(dst + count - SIZE, value);
  }

  LIBC_INLINE static void head_tail(Ptr dst, uint8_t value, size_t count) {
    block(dst, value);
    tail(dst, value, count);
  }

  LIBC_INLINE static void loop_and_tail(Ptr dst, uint8_t value, size_t count) {
    static_assert(SIZE > 1);
    size_t offset = 0;
    do {
      block(dst + offset, value);
      offset += SIZE;
    } while (offset < count - SIZE);
    tail(dst, value, count);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Memmove
///////////////////////////////////////////////////////////////////////////////

template <size_t Size, size_t MaxSize> struct Memmove {
  static_assert(is_power2(MaxSize));
  using T = details::getTypeFor<Size, MaxSize>;
  static constexpr size_t SIZE = Size;

  LIBC_INLINE static void block(Ptr dst, CPtr src) {
    if constexpr (details::is_void_v<T>) {
      deferred_static_assert("Unimplemented Size");
    } else {
      T::store(dst, T::load(src));
    }
  }

  LIBC_INLINE static void head_tail(Ptr dst, CPtr src, size_t count) {
    const size_t offset = count - Size;
    if constexpr (details::is_void_v<T>) {
      deferred_static_assert("Unimplemented Size");
    } else {
      // The load and store operations can be performed in any order as long as
      // they are not interleaved. More investigations are needed to determine
      // the best order.
      const auto head = T::load(src);
      const auto tail = T::load(src + offset);
      T::store(dst, head);
      T::store(dst + offset, tail);
    }
  }

  // Align forward suitable when dst < src. The alignment is performed with
  // an HeadTail operation of count ∈ [Alignment, 2 x Alignment].
  //
  // e.g. Moving two bytes forward, we make sure src is aligned.
  // [  |       |       |       |      ]
  // [____XXXXXXXXXXXXXXXXXXXXXXXXXXXX_]
  // [____LLLLLLLL_____________________]
  // [___________LLLLLLLA______________]
  // [_SSSSSSSS________________________]
  // [________SSSSSSSS_________________]
  //
  // e.g. Moving two bytes forward, we make sure dst is aligned.
  // [  |       |       |       |      ]
  // [____XXXXXXXXXXXXXXXXXXXXXXXXXXXX_]
  // [____LLLLLLLL_____________________]
  // [______LLLLLLLL___________________]
  // [_SSSSSSSS________________________]
  // [___SSSSSSSA______________________]
  template <Arg AlignOn>
  LIBC_INLINE static void align_forward(Ptr &dst, CPtr &src, size_t &count) {
    Ptr prev_dst = dst;
    CPtr prev_src = src;
    size_t prev_count = count;
    align_to_next_boundary<Size, AlignOn>(dst, src, count);
    adjust(Size, dst, src, count);
    head_tail(prev_dst, prev_src, prev_count - count);
  }

  // Align backward suitable when dst > src. The alignment is performed with
  // an HeadTail operation of count ∈ [Alignment, 2 x Alignment].
  //
  // e.g. Moving two bytes backward, we make sure src is aligned.
  // [  |       |       |       |      ]
  // [____XXXXXXXXXXXXXXXXXXXXXXXX_____]
  // [ _________________ALLLLLLL_______]
  // [ ___________________LLLLLLLL_____]
  // [____________________SSSSSSSS_____]
  // [______________________SSSSSSSS___]
  //
  // e.g. Moving two bytes backward, we make sure dst is aligned.
  // [  |       |       |       |      ]
  // [____XXXXXXXXXXXXXXXXXXXXXXXX_____]
  // [ _______________LLLLLLLL_________]
  // [ ___________________LLLLLLLL_____]
  // [__________________ASSSSSSS_______]
  // [______________________SSSSSSSS___]
  template <Arg AlignOn>
  LIBC_INLINE static void align_backward(Ptr &dst, CPtr &src, size_t &count) {
    Ptr headtail_dst = dst + count;
    CPtr headtail_src = src + count;
    size_t headtail_size = 0;
    align_to_next_boundary<Size, AlignOn>(headtail_dst, headtail_src,
                                          headtail_size);
    adjust(-2 * Size, headtail_dst, headtail_src, headtail_size);
    head_tail(headtail_dst, headtail_src, headtail_size);
    count -= headtail_size;
  }

  // Move forward suitable when dst < src. We load the tail bytes before
  // handling the loop.
  //
  // e.g. Moving two bytes
  // [   |       |       |       |       |]
  // [___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
  // [_________________________LLLLLLLL___]
  // [___LLLLLLLL_________________________]
  // [_SSSSSSSS___________________________]
  // [___________LLLLLLLL_________________]
  // [_________SSSSSSSS___________________]
  // [___________________LLLLLLLL_________]
  // [_________________SSSSSSSS___________]
  // [_______________________SSSSSSSS_____]
  LIBC_INLINE static void loop_and_tail_forward(Ptr dst, CPtr src,
                                                size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    const size_t tail_offset = count - Size;
    const auto tail_value = T::load(src + tail_offset);
    size_t offset = 0;
    LIBC_LOOP_NOUNROLL
    do {
      block(dst + offset, src + offset);
      offset += Size;
    } while (offset < count - Size);
    T::store(dst + tail_offset, tail_value);
  }

  // Move backward suitable when dst > src. We load the head bytes before
  // handling the loop.
  //
  // e.g. Moving two bytes
  // [   |       |       |       |       |]
  // [___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
  // [___LLLLLLLL_________________________]
  // [_________________________LLLLLLLL___]
  // [___________________________SSSSSSSS_]
  // [_________________LLLLLLLL___________]
  // [___________________SSSSSSSS_________]
  // [_________LLLLLLLL___________________]
  // [___________SSSSSSSS_________________]
  // [_____SSSSSSSS_______________________]
  LIBC_INLINE static void loop_and_tail_backward(Ptr dst, CPtr src,
                                                 size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    const auto head_value = T::load(src);
    ptrdiff_t offset = count - Size;
    LIBC_LOOP_NOUNROLL
    do {
      block(dst + offset, src + offset);
      offset -= Size;
    } while (offset >= 0);
    T::store(dst, head_value);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Bcmp
///////////////////////////////////////////////////////////////////////////////
template <size_t Size> struct Bcmp {
  static constexpr size_t SIZE = Size;
  static constexpr size_t MaxSize = LLVM_LIBC_IS_DEFINED(LLVM_LIBC_HAS_UINT64)
                                        ? sizeof(uint64_t)
                                        : sizeof(uint32_t);

  template <typename T> LIBC_INLINE static uint32_t load_xor(CPtr p1, CPtr p2) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    return load<T>(p1) ^ load<T>(p2);
  }

  template <typename T>
  LIBC_INLINE static uint32_t load_not_equal(CPtr p1, CPtr p2) {
    return load<T>(p1) != load<T>(p2);
  }

  LIBC_INLINE static BcmpReturnType block(CPtr p1, CPtr p2) {
    if constexpr (Size == 1) {
      return load_xor<uint8_t>(p1, p2);
    } else if constexpr (Size == 2) {
      return load_xor<uint16_t>(p1, p2);
    } else if constexpr (Size == 4) {
      return load_xor<uint32_t>(p1, p2);
    } else if constexpr (Size == 8) {
      return load_not_equal<uint64_t>(p1, p2);
    } else if constexpr (details::useArrayType<Size, MaxSize>()) {
      for (size_t offset = 0; offset < Size; offset += MaxSize)
        if (auto value = Bcmp<MaxSize>::block(p1 + offset, p2 + offset))
          return value;
    } else {
      deferred_static_assert("Unimplemented Size");
    }
    return BcmpReturnType::ZERO();
  }

  LIBC_INLINE static BcmpReturnType tail(CPtr p1, CPtr p2, size_t count) {
    return block(p1 + count - SIZE, p2 + count - SIZE);
  }

  LIBC_INLINE static BcmpReturnType head_tail(CPtr p1, CPtr p2, size_t count) {
    return block(p1, p2) | tail(p1, p2, count);
  }

  LIBC_INLINE static BcmpReturnType loop_and_tail(CPtr p1, CPtr p2,
                                                  size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    size_t offset = 0;
    do {
      if (auto value = block(p1 + offset, p2 + offset))
        return value;
      offset += SIZE;
    } while (offset < count - SIZE);
    return tail(p1, p2, count);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Memcmp
///////////////////////////////////////////////////////////////////////////////
template <size_t Size> struct Memcmp {
  static constexpr size_t SIZE = Size;
  static constexpr size_t MaxSize = LLVM_LIBC_IS_DEFINED(LLVM_LIBC_HAS_UINT64)
                                        ? sizeof(uint64_t)
                                        : sizeof(uint32_t);

  template <typename T> LIBC_INLINE static T load_be(CPtr ptr) {
    return Endian::to_big_endian(load<T>(ptr));
  }

  template <typename T>
  LIBC_INLINE static MemcmpReturnType load_be_diff(CPtr p1, CPtr p2) {
    return load_be<T>(p1) - load_be<T>(p2);
  }

  template <typename T>
  LIBC_INLINE static MemcmpReturnType load_be_cmp(CPtr p1, CPtr p2) {
    const auto la = load_be<T>(p1);
    const auto lb = load_be<T>(p2);
    return la > lb ? 1 : la < lb ? -1 : 0;
  }

  LIBC_INLINE static MemcmpReturnType block(CPtr p1, CPtr p2) {
    if constexpr (Size == 1) {
      return load_be_diff<uint8_t>(p1, p2);
    } else if constexpr (Size == 2) {
      return load_be_diff<uint16_t>(p1, p2);
    } else if constexpr (Size == 4) {
      return load_be_cmp<uint32_t>(p1, p2);
    } else if constexpr (Size == 8) {
      return load_be_cmp<uint64_t>(p1, p2);
    } else if constexpr (details::useArrayType<Size, MaxSize>()) {
      for (size_t offset = 0; offset < Size; offset += MaxSize)
        if (Bcmp<MaxSize>::block(p1 + offset, p2 + offset))
          return Memcmp<MaxSize>::block(p1 + offset, p2 + offset);
      return MemcmpReturnType::ZERO();
    } else if constexpr (Size == 3) {
      if (auto value = Memcmp<2>::block(p1, p2))
        return value;
      return Memcmp<1>::block(p1 + 2, p2 + 2);
    } else {
      deferred_static_assert("Unimplemented Size");
    }
  }

  LIBC_INLINE static MemcmpReturnType tail(CPtr p1, CPtr p2, size_t count) {
    return block(p1 + count - SIZE, p2 + count - SIZE);
  }

  LIBC_INLINE static MemcmpReturnType head_tail(CPtr p1, CPtr p2,
                                                size_t count) {
    if (auto value = block(p1, p2))
      return value;
    return tail(p1, p2, count);
  }

  LIBC_INLINE static MemcmpReturnType loop_and_tail(CPtr p1, CPtr p2,
                                                    size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    size_t offset = 0;
    do {
      if (auto value = block(p1 + offset, p2 + offset))
        return value;
      offset += SIZE;
    } while (offset < count - SIZE);
    return tail(p1, p2, count);
  }
};

} // namespace __llvm_libc::generic

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_GENERIC_H
