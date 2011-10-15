//===------------------------ exception.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#include "exception"

#if __APPLE__
  #include <cxxabi.h>
  using namespace __cxxabiv1;
  using namespace __cxxabiapple;
  // On Darwin, there are two STL shared libraries and a lower level ABI
  // shared libray.  The globals holding the current terminate handler and
  // current unexpected handler are in the ABI library.
  #define __terminate_handler  __cxxabiapple::__cxa_terminate_handler
  #define __unexpected_handler __cxxabiapple::__cxa_unexpected_handler
  #define HAVE_DEPENDENT_EH_ABI 1
#elif defined(LIBCXXRT)
  #include <cxxabi.h>
  using namespace __cxxabiv1;
  #define HAVE_DEPENDENT_EH_ABI 1
#else  // __APPLE__
  static std::terminate_handler  __terminate_handler;
  static std::unexpected_handler __unexpected_handler;
#endif  // __APPLE__

#ifndef LIBCXXRT
// libcxxrt provides implementations of these functions itself.
std::unexpected_handler
std::set_unexpected(std::unexpected_handler func) _NOEXCEPT
{
    return __sync_lock_test_and_set(&__unexpected_handler, func);
}

std::unexpected_handler
std::get_unexpected() _NOEXCEPT
{
    return __sync_fetch_and_add(&__unexpected_handler, (std::unexpected_handler)0);
}

_ATTRIBUTE(noreturn)
void
std::unexpected()
{
    (*std::get_unexpected())();
    // unexpected handler should not return
    std::terminate();
}

std::terminate_handler
std::set_terminate(std::terminate_handler func) _NOEXCEPT
{
    return __sync_lock_test_and_set(&__terminate_handler, func);
}

std::terminate_handler
std::get_terminate() _NOEXCEPT
{
    return __sync_fetch_and_add(&__terminate_handler, (std::terminate_handler)0);
}

_ATTRIBUTE(noreturn)
void
std::terminate() _NOEXCEPT
{
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        (*std::get_terminate())();
        // handler should not return
        ::abort ();
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
        // handler should not throw exception
        ::abort ();
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
}
#endif // LIBCXXRT

bool std::uncaught_exception() _NOEXCEPT
{
#if __APPLE__
    // on Darwin, there is a helper function so __cxa_get_globals is private
    return __cxxabiapple::__cxa_uncaught_exception();
#elif LIBCXXRT
    __cxa_eh_globals * globals = __cxa_get_globals();
    return (globals->uncaughtExceptions != 0);
#else  // __APPLE__
    #warning uncaught_exception not yet implemented
    ::abort();
#endif  // __APPLE__
}

namespace std
{

exception::~exception() _NOEXCEPT
{
}

bad_exception::~bad_exception() _NOEXCEPT
{
}

const char* exception::what() const _NOEXCEPT
{
  return "std::exception";
}

const char* bad_exception::what() const _NOEXCEPT
{
  return "std::bad_exception";
}

exception_ptr::~exception_ptr() _NOEXCEPT
{
#if HAVE_DEPENDENT_EH_ABI
    __cxa_decrement_exception_refcount(__ptr_);
#else
    #warning exception_ptr not yet implemented
    ::abort();
#endif  // __APPLE__
}

exception_ptr::exception_ptr(const exception_ptr& other) _NOEXCEPT
    : __ptr_(other.__ptr_)
{
#if HAVE_DEPENDENT_EH_ABI
    __cxa_increment_exception_refcount(__ptr_);
#else
    #warning exception_ptr not yet implemented
    ::abort();
#endif  // __APPLE__
}

exception_ptr& exception_ptr::operator=(const exception_ptr& other) _NOEXCEPT
{
#if HAVE_DEPENDENT_EH_ABI
    if (__ptr_ != other.__ptr_)
    {
        __cxa_increment_exception_refcount(other.__ptr_);
        __cxa_decrement_exception_refcount(__ptr_);
        __ptr_ = other.__ptr_;
    }
    return *this;
#else  // __APPLE__
    #warning exception_ptr not yet implemented
    ::abort();
#endif  // __APPLE__
}

nested_exception::nested_exception() _NOEXCEPT
    : __ptr_(current_exception())
{
}

nested_exception::~nested_exception() _NOEXCEPT
{
}

_ATTRIBUTE(noreturn)
void
nested_exception::rethrow_nested() const
{
    if (__ptr_ == nullptr)
        terminate();
    rethrow_exception(__ptr_);
}

} // std

std::exception_ptr std::current_exception() _NOEXCEPT
{
#if HAVE_DEPENDENT_EH_ABI
    // be nicer if there was a constructor that took a ptr, then
    // this whole function would be just:
    //    return exception_ptr(__cxa_current_primary_exception());
    std::exception_ptr ptr;
    ptr.__ptr_ = __cxa_current_primary_exception();
    return ptr;
#else  // __APPLE__
    #warning exception_ptr not yet implemented
    ::abort();
#endif  // __APPLE__
}

void std::rethrow_exception(exception_ptr p)
{
#if HAVE_DEPENDENT_EH_ABI
    __cxa_rethrow_primary_exception(p.__ptr_);
    // if p.__ptr_ is NULL, above returns so we terminate
    terminate();
#else  // __APPLE__
    #warning exception_ptr not yet implemented
    ::abort();
#endif  // __APPLE__
}
