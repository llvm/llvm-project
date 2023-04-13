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

namespace __llvm_libc {
// Compiler types using the vector attributes.
using uint8x1_t = uint8_t __attribute__((__vector_size__(1)));
using uint8x2_t = uint8_t __attribute__((__vector_size__(2)));
using uint8x4_t = uint8_t __attribute__((__vector_size__(4)));
using uint8x8_t = uint8_t __attribute__((__vector_size__(8)));
using uint8x16_t = uint8_t __attribute__((__vector_size__(16)));
using uint8x32_t = uint8_t __attribute__((__vector_size__(32)));
using uint8x64_t = uint8_t __attribute__((__vector_size__(64)));
} // namespace __llvm_libc

namespace __llvm_libc::generic {
// We accept three types of values as elements for generic operations:
// - scalar : unsigned integral types
// - vector : compiler types using the vector attributes
// - array  : a cpp::array<T, N> where T is itself either a scalar or a vector.
// The following traits help discriminate between these cases.
template <typename T>
constexpr bool is_scalar_v = cpp::is_integral_v<T> && cpp::is_unsigned_v<T>;

template <typename T>
constexpr bool is_vector_v =
    cpp::details::is_unqualified_any_of<T, uint8x1_t, uint8x2_t, uint8x4_t,
                                        uint8x8_t, uint8x16_t, uint8x32_t,
                                        uint8x64_t>();

template <class T> struct is_array : cpp::false_type {};
template <class T, size_t N> struct is_array<cpp::array<T, N>> {
  static constexpr bool value = is_scalar_v<T> || is_vector_v<T>;
};
template <typename T> constexpr bool is_array_v = is_array<T>::value;

template <typename T>
constexpr bool is_element_type_v =
    is_scalar_v<T> || is_vector_v<T> || is_array_v<T>;

//
template <class T> struct array_size {};
template <class T, size_t N>
struct array_size<cpp::array<T, N>> : cpp::integral_constant<size_t, N> {};
template <typename T> constexpr size_t array_size_v = array_size<T>::value;

// Generic operations for the above type categories.

template <typename T> T load(CPtr src) {
  static_assert(is_element_type_v<T>);
  if constexpr (is_scalar_v<T> || is_vector_v<T>) {
    return ::__llvm_libc::load<T>(src);
  } else if constexpr (is_array_v<T>) {
    using value_type = typename T::value_type;
    T Value;
    for (size_t I = 0; I < array_size_v<T>; ++I)
      Value[I] = load<value_type>(src + (I * sizeof(value_type)));
    return Value;
  }
}

template <typename T> void store(Ptr dst, T value) {
  static_assert(is_element_type_v<T>);
  if constexpr (is_scalar_v<T> || is_vector_v<T>) {
    ::__llvm_libc::store<T>(dst, value);
  } else if constexpr (is_array_v<T>) {
    using value_type = typename T::value_type;
    for (size_t I = 0; I < array_size_v<T>; ++I)
      store<value_type>(dst + (I * sizeof(value_type)), value[I]);
  }
}

template <typename T> T splat(uint8_t value) {
  static_assert(is_scalar_v<T> || is_vector_v<T>);
  if constexpr (is_scalar_v<T>)
    return T(~0) / T(0xFF) * T(value);
  else if constexpr (is_vector_v<T>) {
    T Out;
    // This for loop is optimized out for vector types.
    for (size_t i = 0; i < sizeof(T); ++i)
      Out[i] = value;
    return Out;
  }
}

static_assert((UINTPTR_MAX == 4294967295U) ||
                  (UINTPTR_MAX == 18446744073709551615UL),
              "We currently only support 32- or 64-bit platforms");

#if defined(LIBC_TARGET_ARCH_IS_X86_64) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
#define LLVM_LIBC_HAS_UINT64
#endif

namespace details {
// Checks that each type is sorted in strictly decreasing order of size.
// i.e. sizeof(First) > sizeof(Second) > ... > sizeof(Last)
template <typename First> constexpr bool is_decreasing_size() {
  return sizeof(First) == 1;
}
template <typename First, typename Second, typename... Next>
constexpr bool is_decreasing_size() {
  if constexpr (sizeof...(Next) > 0)
    return sizeof(First) > sizeof(Second) && is_decreasing_size<Next...>();
  else
    return sizeof(First) > sizeof(Second) && is_decreasing_size<Second>();
}

template <size_t Size, typename... Ts> struct Largest;
template <size_t Size> struct Largest<Size> : cpp::type_identity<uint8_t> {};
template <size_t Size, typename T, typename... Ts>
struct Largest<Size, T, Ts...> {
  using next = Largest<Size, Ts...>;
  using type = cpp::conditional_t<(Size >= sizeof(T)), T, typename next::type>;
};

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

template <typename First, typename... Ts> struct SupportedTypes {
  static_assert(details::is_decreasing_size<First, Ts...>());

  using MaxType = First;

  template <size_t Size>
  using TypeFor = typename details::Largest<Size, First, Ts...>::type;
};

// Returns the sum of the sizeof of all the TS types.
template <typename... TS> static constexpr size_t sum_sizeof() {
  return (... + sizeof(TS));
}

// Map from sizes to structures offering static load, store and splat methods.
// Note: On platforms lacking vector support, we use the ArrayType below and
// decompose the operation in smaller pieces.

// Lists a generic native types to use for Memset and Memmove operations.
// TODO: Inject the native types within Memset and Memmove depending on the
// target architectures and derive MaxSize from it.
using NativeTypeMap = SupportedTypes<uint8x64_t, //
                                     uint8x32_t, //
                                     uint8x16_t,
#if defined(LLVM_LIBC_HAS_UINT64)
                                     uint64_t, // Not available on 32bit
#endif
                                     uint32_t, //
                                     uint16_t, //
                                     uint8_t>;

namespace details {

// Helper to test if a type is void.
template <typename T> inline constexpr bool is_void_v = cpp::is_same_v<T, void>;

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
    cpp::array<NativeTypeMap::TypeFor<MaxSize>, Size / MaxSize>,
    NativeTypeMap::TypeFor<Size>>;

} // namespace details

///////////////////////////////////////////////////////////////////////////////
// Memset
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename... TS> struct Memset {
  static constexpr size_t SIZE = sum_sizeof<T, TS...>();

  LIBC_INLINE static void block(Ptr dst, uint8_t value) {
    static_assert(is_element_type_v<T>);
    if constexpr (is_scalar_v<T> || is_vector_v<T>) {
      store<T>(dst, splat<T>(value));
    } else if constexpr (is_array_v<T>) {
      using value_type = typename T::value_type;
      const auto Splat = splat<value_type>(value);
      for (size_t I = 0; I < array_size_v<T>; ++I)
        store<value_type>(dst + (I * sizeof(value_type)), Splat);
    }
    if constexpr (sizeof...(TS))
      Memset<TS...>::block(dst + sizeof(T), value);
  }

  LIBC_INLINE static void tail(Ptr dst, uint8_t value, size_t count) {
    block(dst + count - SIZE, value);
  }

  LIBC_INLINE static void head_tail(Ptr dst, uint8_t value, size_t count) {
    block(dst, value);
    tail(dst, value, count);
  }

  LIBC_INLINE static void loop_and_tail(Ptr dst, uint8_t value, size_t count) {
    static_assert(SIZE > 1, "a loop of size 1 does not need tail");
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

template <typename T> struct Memmove {
  static constexpr size_t SIZE = sum_sizeof<T>();

  LIBC_INLINE static void block(Ptr dst, CPtr src) {
    store<T>(dst, load<T>(src));
  }

  LIBC_INLINE static void head_tail(Ptr dst, CPtr src, size_t count) {
    const size_t offset = count - SIZE;
    // The load and store operations can be performed in any order as long as
    // they are not interleaved. More investigations are needed to determine
    // the best order.
    const auto head = load<T>(src);
    const auto tail = load<T>(src + offset);
    store<T>(dst, head);
    store<T>(dst + offset, tail);
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
    align_to_next_boundary<SIZE, AlignOn>(dst, src, count);
    adjust(SIZE, dst, src, count);
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
    align_to_next_boundary<SIZE, AlignOn>(headtail_dst, headtail_src,
                                          headtail_size);
    adjust(-2 * SIZE, headtail_dst, headtail_src, headtail_size);
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
    static_assert(SIZE > 1, "a loop of size 1 does not need tail");
    const size_t tail_offset = count - SIZE;
    const auto tail_value = load<T>(src + tail_offset);
    size_t offset = 0;
    LIBC_LOOP_NOUNROLL
    do {
      block(dst + offset, src + offset);
      offset += SIZE;
    } while (offset < count - SIZE);
    store<T>(dst + tail_offset, tail_value);
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
    static_assert(SIZE > 1, "a loop of size 1 does not need tail");
    const auto head_value = load<T>(src);
    ptrdiff_t offset = count - SIZE;
    LIBC_LOOP_NOUNROLL
    do {
      block(dst + offset, src + offset);
      offset -= SIZE;
    } while (offset >= 0);
    store<T>(dst, head_value);
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
