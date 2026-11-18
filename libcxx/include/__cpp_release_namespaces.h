// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CPP_RELEASE_NAMESPACES_H
#define _LIBCPP___CPP_RELEASE_NAMESPACES_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// PROTOTYPE: Standard-release namespaces as proposed by P3866R0 (V2: An Evolution Path for the Standard Library).
//
// Each std::cppNN namespace names the most recent version of every component available in C++NN. Since most
// components never get a new version, each release namespace simply makes everything in std visible through a
// using-directive, and only the components that were revised in that release (e.g. `vector` -> `vector2` in C++29)
// are explicitly declared in it, which hides the std entity of the same name from qualified lookup:
//
//    std::cpp29::vector<int>   // finds the alias declared in std::cpp29, i.e. std::vector2<int>
//    std::cpp29::string        // not declared in std::cpp29, found through the using-directive: std::string
//
// Those explicit declarations live next to the component they refer to (e.g. in <__vector/vector2.h>), so that
// std::cpp29::vector is available exactly when std::vector2 is.
//
// Nested namespaces need special care: as soon as a release namespace declares its own `pmr` namespace (to make
// std::cpp29::pmr::vector refer to std::pmr::vector2), that namespace hides std::pmr entirely, so it needs its own
// using-directive as well.

_LIBCPP_BEGIN_NAMESPACE_STD

namespace pmr {} // so we can refer to it below

#if _LIBCPP_STD_VER >= 11
namespace cpp11 {
using namespace std;
} // namespace cpp11
#endif

#if _LIBCPP_STD_VER >= 14
namespace cpp14 {
using namespace std;
} // namespace cpp14
#endif

#if _LIBCPP_STD_VER >= 17
namespace cpp17 {
using namespace std;
} // namespace cpp17
#endif

#if _LIBCPP_STD_VER >= 20
namespace cpp20 {
using namespace std;
} // namespace cpp20
#endif

#if _LIBCPP_STD_VER >= 23
namespace cpp23 {
using namespace std;
} // namespace cpp23
#endif

#if _LIBCPP_STD_VER >= 26
namespace cpp26 {
using namespace std;
} // namespace cpp26
#endif

#if _LIBCPP_STD_VER >= 29
namespace cpp29 {
using namespace std;

namespace pmr {
using namespace std::pmr;
} // namespace pmr
} // namespace cpp29
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CPP_RELEASE_NAMESPACES_H
