//===-- Elementary operations to compose memory primitives ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#include "src/__support/endian.h"
#include "src/string/memory_utils/utils.h"

namespace __llvm_libc {

// Elementary Operations
// --------------------------------
// We define abstract elementary operations acting on fixed chunks of memory.
// These are low level building blocks that are meant to be assembled to compose
// higher order abstractions. Each function is defined twice: once with
// fixed-size operations, and once with runtime-size operations.

// Fixed-size copy from 'src' to 'dst'.
template <typename Element>
void copy(char *__restrict dst, const char *__restrict src) {
  Element::copy(dst, src);
}
// Runtime-size copy from 'src' to 'dst'.
template <typename Element>
void copy(char *__restrict dst, const char *__restrict src, size_t size) {
  Element::copy(dst, src, size);
}

// Fixed-size move from 'src' to 'dst'.
template <typename Element> void move(char *dst, const char *src) {
  Element::move(dst, src);
}
// Runtime-size move from 'src' to 'dst'.
template <typename Element> void move(char *dst, const char *src, size_t size) {
  Element::move(dst, src, size);
}

// Fixed-size equality between 'lhs' and 'rhs'.
template <typename Element> bool equals(const char *lhs, const char *rhs) {
  return Element::equals(lhs, rhs);
}
// Runtime-size equality between 'lhs' and 'rhs'.
template <typename Element>
bool equals(const char *lhs, const char *rhs, size_t size) {
  return Element::equals(lhs, rhs, size);
}

// Fixed-size three-way comparison between 'lhs' and 'rhs'.
template <typename Element>
int three_way_compare(const char *lhs, const char *rhs) {
  return Element::three_way_compare(lhs, rhs);
}
// Runtime-size three-way comparison between 'lhs' and 'rhs'.
template <typename Element>
int three_way_compare(const char *lhs, const char *rhs, size_t size) {
  return Element::three_way_compare(lhs, rhs, size);
}

// Fixed-size initialization.
template <typename Element>
void splat_set(char *dst, const unsigned char value) {
  Element::splat_set(dst, value);
}
// Runtime-size initialization.
template <typename Element>
void splat_set(char *dst, const unsigned char value, size_t size) {
  Element::splat_set(dst, value, size);
}

// Stack placeholder for Move operations.
template <typename Element> struct Storage { char bytes[Element::SIZE]; };

// Fixed-size Higher-Order Operations
// ----------------------------------
// - Repeated<Type, ElementCount>: Repeat the operation several times in a row.
// - Chained<Types...>: Chain the operation of several types.

// Repeat the operation several times in a row.
template <typename Element, size_t ElementCount> struct Repeated {
  static constexpr size_t SIZE = ElementCount * Element::SIZE;

  static void copy(char *__restrict dst, const char *__restrict src) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::SIZE;
      Element::copy(dst + offset, src + offset);
    }
  }

  static void move(char *dst, const char *src) {
    const auto value = Element::load(src);
    Repeated<Element, ElementCount - 1>::move(dst + Element::SIZE,
                                              src + Element::SIZE);
    Element::store(dst, value);
  }

  static bool equals(const char *lhs, const char *rhs) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::SIZE;
      if (!Element::equals(lhs + offset, rhs + offset))
        return false;
    }
    return true;
  }

  static int three_way_compare(const char *lhs, const char *rhs) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::SIZE;
      // We make the assumption that 'equals' is cheaper than
      // 'three_way_compare'.
      if (Element::equals(lhs + offset, rhs + offset))
        continue;
      return Element::three_way_compare(lhs + offset, rhs + offset);
    }
    return 0;
  }

  static void splat_set(char *dst, const unsigned char value) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::SIZE;
      Element::splat_set(dst + offset, value);
    }
  }

  static Storage<Repeated> load(const char *ptr) {
    Storage<Repeated> value;
    copy(reinterpret_cast<char *>(&value), ptr);
    return value;
  }

  static void store(char *ptr, Storage<Repeated> value) {
    copy(ptr, reinterpret_cast<const char *>(&value));
  }
};

template <typename Element> struct Repeated<Element, 0> {
  static void move(char *dst, const char *src) {}
};

// Chain the operation of several types.
// For instance, to handle a 3 bytes operation, one can use:
// Chained<UINT16, UINT8>::Operation();
template <typename... Types> struct Chained;

template <typename Head, typename... Tail> struct Chained<Head, Tail...> {
  static constexpr size_t SIZE = Head::SIZE + Chained<Tail...>::SIZE;

  static void copy(char *__restrict dst, const char *__restrict src) {
    Chained<Tail...>::copy(dst + Head::SIZE, src + Head::SIZE);
    __llvm_libc::copy<Head>(dst, src);
  }

  static void move(char *dst, const char *src) {
    const auto value = Head::load(src);
    Chained<Tail...>::move(dst + Head::SIZE, src + Head::SIZE);
    Head::store(dst, value);
  }

  static bool equals(const char *lhs, const char *rhs) {
    if (!__llvm_libc::equals<Head>(lhs, rhs))
      return false;
    return Chained<Tail...>::equals(lhs + Head::SIZE, rhs + Head::SIZE);
  }

  static int three_way_compare(const char *lhs, const char *rhs) {
    if (__llvm_libc::equals<Head>(lhs, rhs))
      return Chained<Tail...>::three_way_compare(lhs + Head::SIZE,
                                                 rhs + Head::SIZE);
    return __llvm_libc::three_way_compare<Head>(lhs, rhs);
  }

  static void splat_set(char *dst, const unsigned char value) {
    Chained<Tail...>::splat_set(dst + Head::SIZE, value);
    __llvm_libc::splat_set<Head>(dst, value);
  }
};

template <> struct Chained<> {
  static constexpr size_t SIZE = 0;
  static void copy(char *__restrict dst, const char *__restrict src) {}
  static void move(char *dst, const char *src) {}
  static bool equals(const char *lhs, const char *rhs) { return true; }
  static int three_way_compare(const char *lhs, const char *rhs) { return 0; }
  static void splat_set(char *dst, const unsigned char value) {}
};

// Overlap ElementA and ElementB so they span Size bytes.
template <size_t Size, typename ElementA, typename ElementB = ElementA>
struct Overlap {
  static constexpr size_t SIZE = Size;
  static_assert(ElementB::SIZE <= ElementA::SIZE, "ElementB too big");
  static_assert(ElementA::SIZE <= Size, "ElementA too big");
  static_assert((ElementA::SIZE + ElementB::SIZE) >= Size,
                "Elements too small to overlap");
  static constexpr size_t OFFSET = SIZE - ElementB::SIZE;

  static void copy(char *__restrict dst, const char *__restrict src) {
    ElementA::copy(dst, src);
    ElementB::copy(dst + OFFSET, src + OFFSET);
  }

  static void move(char *dst, const char *src) {
    const auto value_a = ElementA::load(src);
    const auto value_b = ElementB::load(src + OFFSET);
    ElementB::store(dst + OFFSET, value_b);
    ElementA::store(dst, value_a);
  }

  static bool equals(const char *lhs, const char *rhs) {
    if (!ElementA::equals(lhs, rhs))
      return false;
    if (!ElementB::equals(lhs + OFFSET, rhs + OFFSET))
      return false;
    return true;
  }

  static int three_way_compare(const char *lhs, const char *rhs) {
    if (!ElementA::equals(lhs, rhs))
      return ElementA::three_way_compare(lhs, rhs);
    if (!ElementB::equals(lhs + OFFSET, rhs + OFFSET))
      return ElementB::three_way_compare(lhs + OFFSET, rhs + OFFSET);
    return 0;
  }

  static void splat_set(char *dst, const unsigned char value) {
    ElementA::splat_set(dst, value);
    ElementB::splat_set(dst + OFFSET, value);
  }
};

// Runtime-size Higher-Order Operations
// ------------------------------------
// - Tail<T>: Perform the operation on the last 'T::SIZE' bytes of the buffer.
// - HeadTail<T>: Perform the operation on the first and last 'T::SIZE' bytes
//   of the buffer.
// - Loop<T>: Perform a loop of fixed-sized operations.

// Perform the operation on the last 'T::SIZE' bytes of the buffer.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [________XXXXXXXX___]
//
// Precondition: `size >= T::SIZE`.
template <typename T> struct Tail {
  static void copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    return T::copy(dst + offset(size), src + offset(size));
  }

  static bool equals(const char *lhs, const char *rhs, size_t size) {
    return T::equals(lhs + offset(size), rhs + offset(size));
  }

  static int three_way_compare(const char *lhs, const char *rhs, size_t size) {
    return T::three_way_compare(lhs + offset(size), rhs + offset(size));
  }

  static void splat_set(char *dst, const unsigned char value, size_t size) {
    return T::splat_set(dst + offset(size), value);
  }

  static size_t offset(size_t size) { return size - T::SIZE; }
};

// Perform the operation on the first and last 'T::SIZE' bytes of the buffer.
// This is useful for overlapping operations.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [__XXXXXXXX_________]
// [________XXXXXXXX___]
//
// Precondition: `size >= T::SIZE && size <= 2 x T::SIZE`.
template <typename T> struct HeadTail {
  static void copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    T::copy(dst, src);
    Tail<T>::copy(dst, src, size);
  }

  static void move(char *dst, const char *src, size_t size) {
    const size_t offset = Tail<T>::offset(size);
    const auto head_value = T::load(src);
    const auto tail_value = T::load(src + offset);
    T::store(dst + offset, tail_value);
    T::store(dst, head_value);
  }

  static bool equals(const char *lhs, const char *rhs, size_t size) {
    if (!T::equals(lhs, rhs))
      return false;
    return Tail<T>::equals(lhs, rhs, size);
  }

  static int three_way_compare(const char *lhs, const char *rhs, size_t size) {
    if (!T::equals(lhs, rhs))
      return T::three_way_compare(lhs, rhs);
    return Tail<T>::three_way_compare(lhs, rhs, size);
  }

  static void splat_set(char *dst, const unsigned char value, size_t size) {
    T::splat_set(dst, value);
    Tail<T>::splat_set(dst, value, size);
  }
};

// Simple loop ending with a Tail operation.
//
// e.g. with
// [12345678123456781234567812345678]
// [__XXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
// [__XXXXXXXX_______________________]
// [__________XXXXXXXX_______________]
// [__________________XXXXXXXX_______]
// [______________________XXXXXXXX___]
//
// Precondition:
// - size >= T::SIZE
template <typename T, typename TailT = T> struct Loop {
  static_assert(T::SIZE == TailT::SIZE,
                "Tail type must have the same size as T");

  static void copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    size_t offset = 0;
    do {
      T::copy(dst + offset, src + offset);
      offset += T::SIZE;
    } while (offset < size - T::SIZE);
    Tail<TailT>::copy(dst, src, size);
  }

  static bool equals(const char *lhs, const char *rhs, size_t size) {
    size_t offset = 0;
    do {
      if (!T::equals(lhs + offset, rhs + offset))
        return false;
      offset += T::SIZE;
    } while (offset < size - T::SIZE);
    return Tail<TailT>::equals(lhs, rhs, size);
  }

  static int three_way_compare(const char *lhs, const char *rhs, size_t size) {
    size_t offset = 0;
    do {
      if (!T::equals(lhs + offset, rhs + offset))
        return T::three_way_compare(lhs + offset, rhs + offset);
      offset += T::SIZE;
    } while (offset < size - T::SIZE);
    return Tail<TailT>::three_way_compare(lhs, rhs, size);
  }

  static void splat_set(char *dst, const unsigned char value, size_t size) {
    size_t offset = 0;
    do {
      T::splat_set(dst + offset, value);
      offset += T::SIZE;
    } while (offset < size - T::SIZE);
    Tail<TailT>::splat_set(dst, value, size);
  }
};

enum class Arg { _1, _2, Dst = _1, Src = _2, Lhs = _1, Rhs = _2 };

namespace internal {

// Provides a specialized bump function that adjusts pointers and size so first
// argument (resp. second argument) gets aligned to Alignment.
// We make sure the compiler knows about the adjusted pointer alignment.
template <Arg arg, size_t Alignment> struct AlignHelper {};

template <size_t Alignment> struct AlignHelper<Arg::_1, Alignment> {
  template <typename T1, typename T2>
  static void bump(T1 *__restrict &p1ref, T2 *__restrict &p2ref, size_t &size) {
    const intptr_t offset = offset_to_next_aligned<Alignment>(p1ref);
    p1ref += offset;
    p2ref += offset;
    size -= offset;
    p1ref = assume_aligned<Alignment>(p1ref);
  }
};

template <size_t Alignment> struct AlignHelper<Arg::_2, Alignment> {
  template <typename T1, typename T2>
  static void bump(T1 *__restrict &p1ref, T2 *__restrict &p2ref, size_t &size) {
    const intptr_t offset = offset_to_next_aligned<Alignment>(p2ref);
    p1ref += offset;
    p2ref += offset;
    size -= offset;
    p2ref = assume_aligned<Alignment>(p2ref);
  }
};

} // namespace internal

// An alignment operation that:
// - executes the 'AlignmentT' operation
// - bumps 'dst' or 'src' (resp. 'lhs' or 'rhs') pointers so that the selected
//   pointer gets aligned, size is decreased accordingly.
// - calls the 'NextT' operation.
//
// e.g. A 16-byte Destination Aligned 32-byte Loop Copy can be written as:
// copy<Align<_16, Arg::Dst>::Then<Loop<_32>>>(dst, src, count);
template <typename AlignmentT, Arg AlignOn = Arg::_1> struct Align {
private:
  static constexpr size_t ALIGNMENT = AlignmentT::SIZE;
  static_assert(ALIGNMENT > 1, "Alignment must be more than 1");
  static_assert(is_power2(ALIGNMENT), "Alignment must be a power of 2");

public:
  template <typename NextT> struct Then {
    static void copy(char *__restrict dst, const char *__restrict src,
                     size_t size) {
      AlignmentT::copy(dst, src);
      internal::AlignHelper<AlignOn, ALIGNMENT>::bump(dst, src, size);
      NextT::copy(dst, src, size);
    }

    static bool equals(const char *lhs, const char *rhs, size_t size) {
      if (!AlignmentT::equals(lhs, rhs))
        return false;
      internal::AlignHelper<AlignOn, ALIGNMENT>::bump(lhs, rhs, size);
      return NextT::equals(lhs, rhs, size);
    }

    static int three_way_compare(const char *lhs, const char *rhs,
                                 size_t size) {
      if (!AlignmentT::equals(lhs, rhs))
        return AlignmentT::three_way_compare(lhs, rhs);
      internal::AlignHelper<AlignOn, ALIGNMENT>::bump(lhs, rhs, size);
      return NextT::three_way_compare(lhs, rhs, size);
    }

    static void splat_set(char *dst, const unsigned char value, size_t size) {
      AlignmentT::splat_set(dst, value);
      char *dummy = nullptr;
      internal::AlignHelper<Arg::_1, ALIGNMENT>::bump(dst, dummy, size);
      NextT::splat_set(dst, value, size);
    }
  };
};

// An operation that allows to skip the specified amount of bytes.
template <ptrdiff_t Bytes> struct Skip {
  template <typename NextT> struct Then {
    static void copy(char *__restrict dst, const char *__restrict src,
                     size_t size) {
      NextT::copy(dst + Bytes, src + Bytes, size - Bytes);
    }

    static void copy(char *__restrict dst, const char *__restrict src) {
      NextT::copy(dst + Bytes, src + Bytes);
    }

    static bool equals(const char *lhs, const char *rhs, size_t size) {
      return NextT::equals(lhs + Bytes, rhs + Bytes, size - Bytes);
    }

    static bool equals(const char *lhs, const char *rhs) {
      return NextT::equals(lhs + Bytes, rhs + Bytes);
    }

    static int three_way_compare(const char *lhs, const char *rhs,
                                 size_t size) {
      return NextT::three_way_compare(lhs + Bytes, rhs + Bytes, size - Bytes);
    }

    static int three_way_compare(const char *lhs, const char *rhs) {
      return NextT::three_way_compare(lhs + Bytes, rhs + Bytes);
    }

    static void splat_set(char *dst, const unsigned char value, size_t size) {
      NextT::splat_set(dst + Bytes, value, size - Bytes);
    }

    static void splat_set(char *dst, const unsigned char value) {
      NextT::splat_set(dst + Bytes, value);
    }
  };
};

// Fixed-size Builtin Operations
// -----------------------------
// Note: Do not use 'builtin' right now as it requires the implementation of the
// `_inline` versions of all the builtins. Theoretically, Clang can still turn
// them into calls to the C library leading to reentrancy problems.
namespace builtin {

#ifndef __has_builtin
#define __has_builtin(x) 0 // Compatibility with non-clang compilers.
#endif

template <size_t Size> struct Builtin {
  static constexpr size_t SIZE = Size;

  static void copy(char *__restrict dst, const char *__restrict src) {
#if LLVM_LIBC_HAVE_MEMORY_SANITIZER || LLVM_LIBC_HAVE_ADDRESS_SANITIZER
    for_loop_copy(dst, src);
#elif __has_builtin(__builtin_memcpy_inline)
    // __builtin_memcpy_inline guarantees to never call external functions.
    // Unfortunately it is not widely available.
    __builtin_memcpy_inline(dst, src, SIZE);
#else
    for_loop_copy(dst, src);
#endif
  }

  static void move(char *dst, const char *src) {
#if LLVM_LIBC_HAVE_MEMORY_SANITIZER || LLVM_LIBC_HAVE_ADDRESS_SANITIZER
    for_loop_move(dst, src);
#elif __has_builtin(__builtin_memmove)
    __builtin_memmove(dst, src, SIZE);
#else
    for_loop_move(dst, src);
#endif
  }

#if __has_builtin(__builtin_memcmp_inline)
#define LLVM_LIBC_MEMCMP __builtin_memcmp_inline
#else
#define LLVM_LIBC_MEMCMP __builtin_memcmp
#endif

  static bool equals(const char *lhs, const char *rhs) {
    return LLVM_LIBC_MEMCMP(lhs, rhs, SIZE) == 0;
  }

  static int three_way_compare(const char *lhs, const char *rhs) {
    return LLVM_LIBC_MEMCMP(lhs, rhs, SIZE);
  }

  static void splat_set(char *dst, const unsigned char value) {
    __builtin_memset(dst, value, SIZE);
  }

private:
  // Copies `SIZE` bytes from `src` to `dst` using a for loop.
  // This code requires the use of `-fno-builtin-memcpy` to prevent the compiler
  // from turning the for-loop back into `__builtin_memcpy`.
  static void for_loop_copy(char *__restrict dst, const char *__restrict src) {
    for (size_t i = 0; i < SIZE; ++i)
      dst[i] = src[i];
  }

  static void for_loop_move(char *dst, const char *src) {
    for (size_t i = 0; i < SIZE; ++i)
      dst[i] = src[i];
  }
};

using _1 = Builtin<1>;
using _2 = Builtin<2>;
using _3 = Builtin<3>;
using _4 = Builtin<4>;
using _8 = Builtin<8>;
using _16 = Builtin<16>;
using _32 = Builtin<32>;
using _64 = Builtin<64>;
using _128 = Builtin<128>;

} // namespace builtin

// Fixed-size Scalar Operations
// ----------------------------
namespace scalar {

// The Scalar type makes use of simple sized integers.
template <typename T> struct Scalar {
  static constexpr size_t SIZE = sizeof(T);

  static void copy(char *__restrict dst, const char *__restrict src) {
    store(dst, load(src));
  }

  static void move(char *dst, const char *src) { store(dst, load(src)); }

  static bool equals(const char *lhs, const char *rhs) {
    return load(lhs) == load(rhs);
  }

  static int three_way_compare(const char *lhs, const char *rhs) {
    return scalar_three_way_compare(load(lhs), load(rhs));
  }

  static void splat_set(char *dst, const unsigned char value) {
    store(dst, get_splatted_value(value));
  }

  static int scalar_three_way_compare(T a, T b);

  static T load(const char *ptr) {
    T value;
    builtin::Builtin<SIZE>::copy(reinterpret_cast<char *>(&value), ptr);
    return value;
  }
  static void store(char *ptr, T value) {
    builtin::Builtin<SIZE>::copy(ptr, reinterpret_cast<const char *>(&value));
  }

private:
  static T get_splatted_value(const unsigned char value) {
    return T(~0) / T(0xFF) * T(value);
  }
};

template <>
inline int Scalar<uint8_t>::scalar_three_way_compare(uint8_t a, uint8_t b) {
  const int16_t la = Endian::to_big_endian(a);
  const int16_t lb = Endian::to_big_endian(b);
  return la - lb;
}
template <>
inline int Scalar<uint16_t>::scalar_three_way_compare(uint16_t a, uint16_t b) {
  const int32_t la = Endian::to_big_endian(a);
  const int32_t lb = Endian::to_big_endian(b);
  return la - lb;
}
template <>
inline int Scalar<uint32_t>::scalar_three_way_compare(uint32_t a, uint32_t b) {
  const uint32_t la = Endian::to_big_endian(a);
  const uint32_t lb = Endian::to_big_endian(b);
  return la > lb ? 1 : la < lb ? -1 : 0;
}
template <>
inline int Scalar<uint64_t>::scalar_three_way_compare(uint64_t a, uint64_t b) {
  const uint64_t la = Endian::to_big_endian(a);
  const uint64_t lb = Endian::to_big_endian(b);
  return la > lb ? 1 : la < lb ? -1 : 0;
}

using UINT8 = Scalar<uint8_t>;   // 1 Byte
using UINT16 = Scalar<uint16_t>; // 2 Bytes
using UINT32 = Scalar<uint32_t>; // 4 Bytes
using UINT64 = Scalar<uint64_t>; // 8 Bytes

using _1 = UINT8;
using _2 = UINT16;
using _3 = Chained<UINT16, UINT8>;
using _4 = UINT32;
using _8 = UINT64;
using _16 = Repeated<_8, 2>;
using _32 = Repeated<_8, 4>;
using _64 = Repeated<_8, 8>;
using _128 = Repeated<_8, 16>;

} // namespace scalar
} // namespace __llvm_libc

#include <src/string/memory_utils/elements_aarch64.h>
#include <src/string/memory_utils/elements_x86.h>

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_H
