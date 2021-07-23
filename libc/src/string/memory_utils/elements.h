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

// Fixed-size copies from 'src' to 'dst'.
template <typename Element>
void Copy(char *__restrict dst, const char *__restrict src) {
  Element::Copy(dst, src);
}
// Runtime-size copies from 'src' to 'dst'.
template <typename Element>
void Copy(char *__restrict dst, const char *__restrict src, size_t size) {
  Element::Copy(dst, src, size);
}

// Fixed-size equality between 'lhs' and 'rhs'.
template <typename Element> bool Equals(const char *lhs, const char *rhs) {
  return Element::Equals(lhs, rhs);
}
// Runtime-size equality between 'lhs' and 'rhs'.
template <typename Element>
bool Equals(const char *lhs, const char *rhs, size_t size) {
  return Element::Equals(lhs, rhs, size);
}

// Fixed-size three-way comparison between 'lhs' and 'rhs'.
template <typename Element>
int ThreeWayCompare(const char *lhs, const char *rhs) {
  return Element::ThreeWayCompare(lhs, rhs);
}
// Runtime-size three-way comparison between 'lhs' and 'rhs'.
template <typename Element>
int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
  return Element::ThreeWayCompare(lhs, rhs, size);
}

// Fixed-size initialization.
template <typename Element>
void SplatSet(char *dst, const unsigned char value) {
  Element::SplatSet(dst, value);
}
// Runtime-size initialization.
template <typename Element>
void SplatSet(char *dst, const unsigned char value, size_t size) {
  Element::SplatSet(dst, value, size);
}

// Fixed-size Higher-Order Operations
// ----------------------------------
// - Repeated<Type, ElementCount>: Repeat the operation several times in a row.
// - Chained<Types...>: Chain the operation of several types.

// Repeat the operation several times in a row.
template <typename Element, size_t ElementCount> struct Repeated {
  static constexpr size_t kSize = ElementCount * Element::kSize;

  static void Copy(char *__restrict dst, const char *__restrict src) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::kSize;
      Element::Copy(dst + offset, src + offset);
    }
  }

  static bool Equals(const char *lhs, const char *rhs) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::kSize;
      if (!Element::Equals(lhs + offset, rhs + offset))
        return false;
    }
    return true;
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::kSize;
      // We make the assumption that 'Equals' si cheaper than 'ThreeWayCompare'.
      if (Element::Equals(lhs + offset, rhs + offset))
        continue;
      return Element::ThreeWayCompare(lhs + offset, rhs + offset);
    }
    return 0;
  }

  static void SplatSet(char *dst, const unsigned char value) {
    for (size_t i = 0; i < ElementCount; ++i) {
      const size_t offset = i * Element::kSize;
      Element::SplatSet(dst + offset, value);
    }
  }
};

// Chain the operation of several types.
// For instance, to handle a 3 bytes operation, one can use:
// Chained<UINT16, UINT8>::Operation();
template <typename... Types> struct Chained;

template <typename Head, typename... Tail> struct Chained<Head, Tail...> {
  static constexpr size_t kSize = Head::kSize + Chained<Tail...>::kSize;

  static void Copy(char *__restrict dst, const char *__restrict src) {
    Chained<Tail...>::Copy(dst + Head::kSize, src + Head::kSize);
    __llvm_libc::Copy<Head>(dst, src);
  }

  static bool Equals(const char *lhs, const char *rhs) {
    if (!__llvm_libc::Equals<Head>(lhs, rhs))
      return false;
    return Chained<Tail...>::Equals(lhs + Head::kSize, rhs + Head::kSize);
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    if (__llvm_libc::Equals<Head>(lhs, rhs))
      return Chained<Tail...>::ThreeWayCompare(lhs + Head::kSize,
                                               rhs + Head::kSize);
    return __llvm_libc::ThreeWayCompare<Head>(lhs, rhs);
  }

  static void SplatSet(char *dst, const unsigned char value) {
    Chained<Tail...>::SplatSet(dst + Head::kSize, value);
    __llvm_libc::SplatSet<Head>(dst, value);
  }
};

template <> struct Chained<> {
  static constexpr size_t kSize = 0;
  static void Copy(char *__restrict dst, const char *__restrict src) {}
  static bool Equals(const char *lhs, const char *rhs) { return true; }
  static int ThreeWayCompare(const char *lhs, const char *rhs) { return 0; }
  static void SplatSet(char *dst, const unsigned char value) {}
};

// Runtime-size Higher-Order Operations
// ------------------------------------
// - Tail<T>: Perform the operation on the last 'T::kSize' bytes of the buffer.
// - HeadTail<T>: Perform the operation on the first and last 'T::kSize' bytes
//   of the buffer.
// - Loop<T>: Perform a loop of fixed-sized operations.

// Perform the operation on the last 'T::kSize' bytes of the buffer.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [________XXXXXXXX___]
//
// Precondition: `size >= T::kSize`.
template <typename T> struct Tail {
  static void Copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    return T::Copy(dst + offset(size), src + offset(size));
  }

  static bool Equals(const char *lhs, const char *rhs, size_t size) {
    return T::Equals(lhs + offset(size), rhs + offset(size));
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
    return T::ThreeWayCompare(lhs + offset(size), rhs + offset(size));
  }

  static void SplatSet(char *dst, const unsigned char value, size_t size) {
    return T::SplatSet(dst + offset(size), value);
  }

  static size_t offset(size_t size) { return size - T::kSize; }
};

// Perform the operation on the first and last 'T::kSize' bytes of the buffer.
// This is useful for overlapping operations.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [__XXXXXXXX_________]
// [________XXXXXXXX___]
//
// Precondition: `size >= T::kSize && size <= 2 x T::kSize`.
template <typename T> struct HeadTail {
  static void Copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    T::Copy(dst, src);
    Tail<T>::Copy(dst, src, size);
  }

  static bool Equals(const char *lhs, const char *rhs, size_t size) {
    if (!T::Equals(lhs, rhs))
      return false;
    return Tail<T>::Equals(lhs, rhs, size);
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
    if (!T::Equals(lhs, rhs))
      return T::ThreeWayCompare(lhs, rhs);
    return Tail<T>::ThreeWayCompare(lhs, rhs, size);
  }

  static void SplatSet(char *dst, const unsigned char value, size_t size) {
    T::SplatSet(dst, value);
    Tail<T>::SplatSet(dst, value, size);
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
// - size >= T::kSize
template <typename T> struct Loop {
  static void Copy(char *__restrict dst, const char *__restrict src,
                   size_t size) {
    for (size_t offset = 0; offset < size - T::kSize; offset += T::kSize)
      T::Copy(dst + offset, src + offset);
    Tail<T>::Copy(dst, src, size);
  }

  static bool Equals(const char *lhs, const char *rhs, size_t size) {
    for (size_t offset = 0; offset < size - T::kSize; offset += T::kSize)
      if (!T::Equals(lhs + offset, rhs + offset))
        return false;
    return Tail<T>::Equals(lhs, rhs, size);
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
    for (size_t offset = 0; offset < size - T::kSize; offset += T::kSize)
      if (!T::Equals(lhs + offset, rhs + offset))
        return T::ThreeWayCompare(lhs + offset, rhs + offset);
    return Tail<T>::ThreeWayCompare(lhs, rhs, size);
  }

  static void SplatSet(char *dst, const unsigned char value, size_t size) {
    for (size_t offset = 0; offset < size - T::kSize; offset += T::kSize)
      T::SplatSet(dst + offset, value);
    Tail<T>::SplatSet(dst, value, size);
  }
};

enum class Arg { _1, _2, Dst = _1, Src = _2, Lhs = _1, Rhs = _2 };

namespace internal {

// Provides a specialized Bump function that adjusts pointers and size so first
// argument (resp. second argument) gets aligned to Alignment.
// We make sure the compiler knows about the adjusted pointer alignment.
template <Arg arg, size_t Alignment> struct AlignHelper {};

template <size_t Alignment> struct AlignHelper<Arg::_1, Alignment> {
  template <typename T1, typename T2>
  static void Bump(T1 *__restrict &p1ref, T2 *__restrict &p2ref, size_t &size) {
    const intptr_t offset = offset_to_next_aligned<Alignment>(p1ref);
    p1ref += offset;
    p2ref += offset;
    size -= offset;
    p1ref = assume_aligned<Alignment>(p1ref);
  }
};

template <size_t Alignment> struct AlignHelper<Arg::_2, Alignment> {
  template <typename T1, typename T2>
  static void Bump(T1 *__restrict &p1ref, T2 *__restrict &p2ref, size_t &size) {
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
// Copy<Align<_16, Arg::Dst>::Then<Loop<_32>>>(dst, src, count);
template <typename AlignmentT, Arg AlignOn = Arg::_1> struct Align {
private:
  static constexpr size_t Alignment = AlignmentT::kSize;
  static_assert(Alignment > 1, "Alignment must be more than 1");
  static_assert(is_power2(Alignment), "Alignment must be a power of 2");

public:
  template <typename NextT> struct Then {
    static void Copy(char *__restrict dst, const char *__restrict src,
                     size_t size) {
      AlignmentT::Copy(dst, src);
      internal::AlignHelper<AlignOn, Alignment>::Bump(dst, src, size);
      NextT::Copy(dst, src, size);
    }

    static bool Equals(const char *lhs, const char *rhs, size_t size) {
      if (!AlignmentT::Equals(lhs, rhs))
        return false;
      internal::AlignHelper<AlignOn, Alignment>::Bump(lhs, rhs, size);
      return NextT::Equals(lhs, rhs, size);
    }

    static int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
      if (!AlignmentT::Equals(lhs, rhs))
        return AlignmentT::ThreeWayCompare(lhs, rhs);
      internal::AlignHelper<AlignOn, Alignment>::Bump(lhs, rhs, size);
      return NextT::ThreeWayCompare(lhs, rhs, size);
    }

    static void SplatSet(char *dst, const unsigned char value, size_t size) {
      AlignmentT::SplatSet(dst, value);
      char *dummy = nullptr;
      internal::AlignHelper<Arg::_1, Alignment>::Bump(dst, dummy, size);
      NextT::SplatSet(dst, value, size);
    }
  };
};

// An operation that allows to skip the specified amount of bytes.
template <ptrdiff_t Bytes> struct Skip {
  template <typename NextT> struct Then {
    static void Copy(char *__restrict dst, const char *__restrict src,
                     size_t size) {
      NextT::Copy(dst + Bytes, src + Bytes, size - Bytes);
    }

    static void Copy(char *__restrict dst, const char *__restrict src) {
      NextT::Copy(dst + Bytes, src + Bytes);
    }

    static bool Equals(const char *lhs, const char *rhs, size_t size) {
      return NextT::Equals(lhs + Bytes, rhs + Bytes, size - Bytes);
    }

    static bool Equals(const char *lhs, const char *rhs) {
      return NextT::Equals(lhs + Bytes, rhs + Bytes);
    }

    static int ThreeWayCompare(const char *lhs, const char *rhs, size_t size) {
      return NextT::ThreeWayCompare(lhs + Bytes, rhs + Bytes, size - Bytes);
    }

    static int ThreeWayCompare(const char *lhs, const char *rhs) {
      return NextT::ThreeWayCompare(lhs + Bytes, rhs + Bytes);
    }

    static void SplatSet(char *dst, const unsigned char value, size_t size) {
      NextT::SplatSet(dst + Bytes, value, size - Bytes);
    }

    static void SplatSet(char *dst, const unsigned char value) {
      NextT::SplatSet(dst + Bytes, value);
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
  static constexpr size_t kSize = Size;

  static void Copy(char *__restrict dst, const char *__restrict src) {
#if LLVM_LIBC_HAVE_MEMORY_SANITIZER || LLVM_LIBC_HAVE_ADDRESS_SANITIZER
    ForLoopCopy(dst, src);
#elif __has_builtin(__builtin_memcpy_inline)
    // __builtin_memcpy_inline guarantees to never call external functions.
    // Unfortunately it is not widely available.
    __builtin_memcpy_inline(dst, src, kSize);
#elif __has_builtin(__builtin_memcpy)
    __builtin_memcpy(dst, src, kSize);
#else
    ForLoopCopy(dst, src);
#endif
  }

#if __has_builtin(__builtin_memcmp_inline)
#define LLVM_LIBC_MEMCMP __builtin_memcmp_inline
#else
#define LLVM_LIBC_MEMCMP __builtin_memcmp
#endif

  static bool Equals(const char *lhs, const char *rhs) {
    return LLVM_LIBC_MEMCMP(lhs, rhs, kSize) == 0;
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    return LLVM_LIBC_MEMCMP(lhs, rhs, kSize);
  }

  static void SplatSet(char *dst, const unsigned char value) {
    __builtin_memset(dst, value, kSize);
  }

private:
  // Copies `kSize` bytes from `src` to `dst` using a for loop.
  // This code requires the use of `-fno-buitin-memcpy` to prevent the compiler
  // from turning the for-loop back into `__builtin_memcpy`.
  static void ForLoopCopy(char *__restrict dst, const char *__restrict src) {
    for (size_t i = 0; i < kSize; ++i)
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
  static constexpr size_t kSize = sizeof(T);

  static void Copy(char *__restrict dst, const char *__restrict src) {
    Store(dst, Load(src));
  }

  static bool Equals(const char *lhs, const char *rhs) {
    return Load(lhs) == Load(rhs);
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    return ScalarThreeWayCompare(Load(lhs), Load(rhs));
  }

  static void SplatSet(char *dst, const unsigned char value) {
    Store(dst, GetSplattedValue(value));
  }

  static int ScalarThreeWayCompare(T a, T b);

private:
  static T Load(const char *ptr) {
    T value;
    builtin::Builtin<kSize>::Copy(reinterpret_cast<char *>(&value), ptr);
    return value;
  }
  static void Store(char *ptr, T value) {
    builtin::Builtin<kSize>::Copy(ptr, reinterpret_cast<const char *>(&value));
  }
  static T GetSplattedValue(const unsigned char value) {
    return T(~0) / T(0xFF) * T(value);
  }
};

template <>
inline int Scalar<uint8_t>::ScalarThreeWayCompare(uint8_t a, uint8_t b) {
  const int16_t la = Endian::ToBigEndian(a);
  const int16_t lb = Endian::ToBigEndian(b);
  return la - lb;
}
template <>
inline int Scalar<uint16_t>::ScalarThreeWayCompare(uint16_t a, uint16_t b) {
  const int32_t la = Endian::ToBigEndian(a);
  const int32_t lb = Endian::ToBigEndian(b);
  return la - lb;
}
template <>
inline int Scalar<uint32_t>::ScalarThreeWayCompare(uint32_t a, uint32_t b) {
  const uint32_t la = Endian::ToBigEndian(a);
  const uint32_t lb = Endian::ToBigEndian(b);
  return la > lb ? 1 : la < lb ? -1 : 0;
}
template <>
inline int Scalar<uint64_t>::ScalarThreeWayCompare(uint64_t a, uint64_t b) {
  const uint64_t la = Endian::ToBigEndian(a);
  const uint64_t lb = Endian::ToBigEndian(b);
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
