//===---------------------------- exception.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_BUILDING_LIBRARY
#include <new>
#include <exception>

namespace std
{

// exception

exception::~exception() _NOEXCEPT
{
}

const char* exception::what() const _NOEXCEPT
{
  return "std::exception";
}

// bad_exception

bad_exception::~bad_exception() _NOEXCEPT
{
}

const char* bad_exception::what() const _NOEXCEPT
{
  return "std::bad_exception";
}


//  bad_alloc

bad_alloc::bad_alloc() _NOEXCEPT
{
}

bad_alloc::~bad_alloc() _NOEXCEPT
{
}

const char*
bad_alloc::what() const _NOEXCEPT
{
    return "std::bad_alloc";
}

// bad_array_new_length

bad_array_new_length::bad_array_new_length() _NOEXCEPT
{
}

bad_array_new_length::~bad_array_new_length() _NOEXCEPT
{
}

const char*
bad_array_new_length::what() const _NOEXCEPT
{
    return "bad_array_new_length";
}

// bad_array_length

#ifndef _LIBCPP_BAD_ARRAY_LENGTH_DEFINED

class _LIBCPP_EXCEPTION_ABI bad_array_length
    : public bad_alloc
{
public:
    bad_array_length() _NOEXCEPT;
    virtual ~bad_array_length() _NOEXCEPT;
    virtual const char* what() const _NOEXCEPT;
};

#endif  // _LIBCPP_BAD_ARRAY_LENGTH_DEFINED

bad_array_length::bad_array_length() _NOEXCEPT
{
}

bad_array_length::~bad_array_length() _NOEXCEPT
{
}

const char*
bad_array_length::what() const _NOEXCEPT
{
    return "bad_array_length";
}


}  // std
