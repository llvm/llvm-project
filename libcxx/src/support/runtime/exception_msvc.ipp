// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_ABI_MICROSOFT
#error this header can only be used when targeting the MSVC ABI
#endif

#include <stdio.h>
#include <stdlib.h>
#include <eh.h>
#include <corecrt_terminate.h>

namespace std {

// libcxxrt provides implementations of these functions itself.
unexpected_handler
set_unexpected(unexpected_handler func) _NOEXCEPT {
  return ::set_unexpected(func);
}

unexpected_handler get_unexpected() _NOEXCEPT {
  return ::_get_unexpected();
}

_LIBCPP_NORETURN
void unexpected() {
    (*get_unexpected())();
    // unexpected handler should not return
    terminate();
}

terminate_handler set_terminate(terminate_handler func) _NOEXCEPT {
  return ::set_terminate(func);
}

terminate_handler get_terminate() _NOEXCEPT {
  return ::_get_terminate();
}

_LIBCPP_NORETURN
void terminate() _NOEXCEPT
{
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        (*get_terminate())();
        // handler should not return
        fprintf(stderr, "terminate_handler unexpectedly returned\n");
        ::abort();
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
        // handler should not throw exception
        fprintf(stderr, "terminate_handler unexpectedly threw an exception\n");
        ::abort();
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
}

bool uncaught_exception() _NOEXCEPT { return uncaught_exceptions() > 0; }

int uncaught_exceptions() _NOEXCEPT {
    return __uncaught_exceptions();
}

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

} // namespace std
