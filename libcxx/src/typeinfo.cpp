//===------------------------- typeinfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "typeinfo"

#if defined(_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY)
std::type_info::~type_info()
{
}
#endif
