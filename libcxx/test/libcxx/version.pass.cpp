// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Test the _LIBCPP_VERSION and _LIBCPP_LIBRARY_VERSION macros

#include <__config>

#ifndef _LIBCPP_VERSION
#error _LIBCPP_VERSION must be defined
#endif

#ifndef _LIBCPP_LIBRARY_VERSION
#error _LIBCPP_LIBRARY_VERSION must be defined
#endif

#include <cassert>

int main() {
  assert(_LIBCPP_VERSION == _LIBCPP_LIBRARY_VERSION);
  assert(std::__libcpp_library_version);
  assert(_LIBCPP_LIBRARY_VERSION == std::__libcpp_library_version());
}
