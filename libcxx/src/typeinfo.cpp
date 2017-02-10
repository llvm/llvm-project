//===------------------------- typeinfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "typeinfo"

// FIXME: Remove __APPLE__ default here once buildit is gone.
#if (!defined(_LIBCPP_ABI_MICROSOFT) && !defined(LIBCXX_BUILDING_LIBCXXABI) && \
    !defined(LIBCXXRT) && !defined(__GLIBCXX__) && \
    !defined(__APPLE__)) || \
    defined(_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY) // FIXME: remove this configuration.
std::type_info::~type_info()
{
}
#endif
