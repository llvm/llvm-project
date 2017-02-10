//===------------------------- typeinfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#if !defined(_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY) && \
    (defined(__APPLE__) || defined(LIBCXXRT) || defined(LIBCXX_BUILDING_LIBCXXABI))
#include <cxxabi.h>
#endif

#include "typeinfo"

#if defined(_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY) || \
    defined(_LIBCPP_ABI_MICROSOFT) // FIXME: This is a temporary workaround
std::type_info::~type_info()
{
}
#endif

#if !defined(LIBCXXRT) && !defined(_LIBCPPABI_VERSION)

std::bad_cast::bad_cast() _NOEXCEPT
{
}

std::bad_typeid::bad_typeid() _NOEXCEPT
{
}

#ifndef __GLIBCXX__

std::bad_cast::~bad_cast() _NOEXCEPT
{
}

const char*
std::bad_cast::what() const _NOEXCEPT
{
  return "std::bad_cast";
}

std::bad_typeid::~bad_typeid() _NOEXCEPT
{
}

const char*
std::bad_typeid::what() const _NOEXCEPT
{
  return "std::bad_typeid";
}

#if defined(__APPLE__) && !defined(_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY)
  // On Darwin, the cxa_bad_* functions cannot be in the lower level library
  // because bad_cast and bad_typeid are defined in his higher level library
  void __cxxabiv1::__cxa_bad_typeid()
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
     throw std::bad_typeid();
#else
     _VSTD::abort();
#endif
  }
  void __cxxabiv1::__cxa_bad_cast()
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
      throw std::bad_cast();
#else
      _VSTD::abort();
#endif
  }
#endif

#endif  // !__GLIBCXX__
#endif  // !LIBCXXRT && !_LIBCPPABI_VERSION
