//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test the contents of <cstddef>

// namespace std {
//   using ptrdiff_t = see below;
//   using size_t = see below;
//   using max_align_t = see below;
//   using nullptr_t = decltype(nullptr);
//
//   enum class byte : unsigned char {};
//
//   // [support.types.byteops], byte type operations
//      [...] other byte-related functionality is tested elsewhere
// }
//
// #define NULL see below
// #define offsetof(P, D) see below

#include <cstddef>
#include "test_macros.h"

using PtrdiffT = std::ptrdiff_t;
using SizeT = std::size_t;
#if TEST_STD_VER >= 11
using MaxAlignT = std::max_align_t;
using NullptrT = std::nullptr_t;
#endif

#if TEST_STD_VER >= 17
using Byte = std::byte;
#endif

#ifndef NULL
#   error "NULL should be defined by <cstddef>"
#endif

#ifndef offsetof
#   error "offsetof() should be defined by <cstddef>"
#endif
