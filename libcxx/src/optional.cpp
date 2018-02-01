//===------------------------ optional.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "optional"

namespace std
{

bad_optional_access::~bad_optional_access() _NOEXCEPT = default;

const char* bad_optional_access::what() const _NOEXCEPT {
  return "bad_optional_access";
  }

} // std

