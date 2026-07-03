//===-- include/flang/Evaluate/object-sizes.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_OBJECT_SIZES_H_
#define FORTRAN_EVALUATE_OBJECT_SIZES_H_

#include <cstddef>

// Object size/alignment for the opaque facades RealValue / CharacterValue and
// their variant-backed implementations RealValueImpl / CharacterValueImpl.
//
// When not cross-compiling, flang-evaluate-object-size-probe measures these
// with the very toolchain (and per build configuration) used for the build and
// emits flang/Evaluate/object-sizes-generated.h into the build tree's include
// directory.  We prefer those measured values whenever that header is available
// on the include path, regardless of -I ordering.  The constants below are the
// fallback used otherwise -- in particular when cross-compiling, where the
// probe cannot run on the build host.  They are verified against the
// implementation classes by static_asserts in real-value.cpp and
// character-value.cpp.
//
// The probe itself (object-size-probe.cpp) compiles with
// FORTRAN_EVALUATE_OBJECT_SIZE_PROBE defined: it generates the header, so it
// must not depend on it.  The dedicated #if branch below omits __has_include so
// dependency scanners do not record the generated header (probe -> generated
// header -> probe cycle).
#if defined(FORTRAN_EVALUATE_OBJECT_SIZE_PROBE)

#error When probing for the object size, must not rely on the default/fallback

#elif __has_include(<flang/Evaluate/object-sizes-generated.h>)
#include <flang/Evaluate/object-sizes-generated.h>
#else

namespace Fortran::evaluate::value::detail {

// These fallbacks assume a 64-bit (LP64/LLP64) host, which covers the targets
// flang is built for (x86_64, AArch64, PowerPC64).  The native build measures
// the real values via object-sizes-generated.h; this is only reached when
// cross-compiling.

// RealValueImpl is a std::variant over flang's fixed-width value::Real types,
// whose widest alternative is a 16-byte, 16-byte-aligned Integer<>.  Because
// all the alternatives are PODs, its layout depends on neither the
// architecture, the C++ standard library, nor its debug/hardening mode: it is
// 32/16 for gcc, clang and MSVC on x86_64/AArch64/PowerPC, in release as well
// as in _GLIBCXX_DEBUG, libc++ hardening and MSVC _ITERATOR_DEBUG_LEVEL builds.
inline constexpr std::size_t kRealObjectSize{32};
inline constexpr std::size_t kRealObjectAlign{16};

// CharacterValueImpl is a std::variant over std::string / std::u16string /
// std::u32string, so its size is governed solely by sizeof(std::string).  This
// is determined by the standard library (and its configuration), not the
// architecture:
//   * libc++:            sizeof(std::string)==24  => variant 32
//   * libstdc++:         sizeof(std::string)==32  => variant 40  (unchanged by
//                        _GLIBCXX_ASSERTIONS or _GLIBCXX_DEBUG)
//   * MSVC STL, release: sizeof(std::string)==32  => variant 40
//   * MSVC STL, debug:   _ITERATOR_DEBUG_LEVEL==2 adds a container-proxy
//   pointer
//                        => variant 48
// libc++ hardening modes do not change the layout.
#if defined(_LIBCPP_VERSION)
inline constexpr std::size_t kCharacterObjectSize{32};
#elif defined(_MSC_VER) && \
    ((defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL >= 2) || \
        (!defined(_ITERATOR_DEBUG_LEVEL) && defined(_DEBUG)))
inline constexpr std::size_t kCharacterObjectSize{48};
#else
inline constexpr std::size_t kCharacterObjectSize{40};
#endif
inline constexpr std::size_t kCharacterObjectAlign{8};

} // namespace Fortran::evaluate::value::detail

#endif // defined(FORTRAN_EVALUATE_OBJECT_SIZE_PROBE) / __has_include
#endif // FORTRAN_EVALUATE_OBJECT_SIZES_H_
