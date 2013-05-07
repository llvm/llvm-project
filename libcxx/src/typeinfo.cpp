//===------------------------- typeinfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#ifndef __has_include
#define __has_include(inc) 0
#endif

#ifdef __APPLE__
#include <cxxabi.h>
#elif defined(LIBCXXRT) || __has_include(<cxxabi.h>)
#include <cxxabi.h>
#endif

#include "typeinfo"

#if !(defined(_LIBCPPABI_VERSION) || defined(LIBCXXRT))

std::bad_cast::bad_cast() _NOEXCEPT
{
}

std::bad_cast::~bad_cast() _NOEXCEPT
{
}

const char*
std::bad_cast::what() const _NOEXCEPT
{
  return "std::bad_cast";
}

std::bad_typeid::bad_typeid() _NOEXCEPT
{
}

std::bad_typeid::~bad_typeid() _NOEXCEPT
{
}

const char*
std::bad_typeid::what() const _NOEXCEPT
{
  return "std::bad_typeid";
}

#ifdef __APPLE__
  // On Darwin, the cxa_bad_* functions cannot be in the lower level library
  // because bad_cast and bad_typeid are defined in his higher level library
  void __cxxabiv1::__cxa_bad_typeid()
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
     throw std::bad_typeid();
#endif
  }
  void __cxxabiv1::__cxa_bad_cast()
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
      throw std::bad_cast();
#endif
  }
#endif

#endif  // _LIBCPPABI_VERSION
