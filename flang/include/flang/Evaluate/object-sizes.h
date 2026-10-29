//===-- include/flang/Evaluate/object-sizes.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Object size/alignment for the opaque facades IntegerValue, RealValue,
// CharacterValue and their variant-backed implementations IntegerValueImpl,
// RealValueImpl, CharacterValueImpl.
//
// When not cross-compiling, flang-evaluate-object-size-probe measures these
// with the very toolchain (and per build configuration) used for the build and
// emits object-sizes-generated.h into the build tree's include
// directory. Those values directly measured are preferred whenever that header
// is available on the include path, regardless of -I ordering. The constants
// below are the fallback used otherwise -- in particular when cross-compiling,
// where the probe cannot run on the build host.  They are verified against the
// implementation classes by static_asserts in integer-value.cpp, real-value.cpp
// and character-value.cpp.
//
// The probe itself (object-size-probe.cpp) compiles with
// FLANG_OBJECT_SIZE_PROBE defined: it generates the header, so it
// must not depend on it.  The dedicated #if branch below omits __has_include so
// dependency scanners do not record the generated header (probe -> generated
// header -> probe cycle).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_OBJECT_SIZES_H_
#define FORTRAN_EVALUATE_OBJECT_SIZES_H_

#include <cstddef>

#ifdef FLANG_OBJECT_SIZE_PROBE
#error This header must not be included into the object-size-probe executable itself (in particular, integer-value-impl.h, real-value-impl.h, character-value-impl.h); it would cause a dependency cycle in incremental builds.
#endif

#if __has_include(<flang/Evaluate/object-sizes-generated.h>)
// Measured object sizes
#include <flang/Evaluate/object-sizes-generated.h>
#else
// Fallback known object sizes
//
// These fallbacks assume a 64-bit (LP64/LLP64) host, which covers the targets
// flang is built for (x86_64, AArch64, PowerPC64).
namespace Fortran::evaluate::value::detail {

inline constexpr std::size_t kIntegerObjectSize{20};
inline constexpr std::size_t kIntegerObjectAlign{4};

inline constexpr std::size_t kRealObjectSize{32};
inline constexpr std::size_t kRealObjectAlign{16};

// CharacterValueImpl is a
// std::variant<std::string, std::u16string, std::u32string>.
//
//  * MSVC STL:  48 bytes with _ITERATOR_DEBUG_LEVEL==2
//               40 bytes otherwise
//  * libc++:    32 bytes
//               invariant to _LIBCPP_HARDENING_MODE
//  * libstdc++: 40 bytes
//               invariant to _GLIBCXX_ASSERTIONS or _GLIBCXX_DEBUG
#if defined(_MSC_VER)
#if ((defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL >= 2) || \
    (!defined(_ITERATOR_DEBUG_LEVEL) && defined(_DEBUG)))
inline constexpr std::size_t kCharacterObjectSize{48};
#else
inline constexpr std::size_t kCharacterObjectSize{40};
#endif
#elif defined(_LIBCPP_VERSION)
inline constexpr std::size_t kCharacterObjectSize{32};
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
inline constexpr std::size_t kCharacterObjectSize{40};
#else
#error Unknown STL implementation
#endif
inline constexpr std::size_t kCharacterObjectAlign{8};

} // namespace Fortran::evaluate::value::detail
#endif

#endif // FORTRAN_EVALUATE_OBJECT_SIZES_H_
