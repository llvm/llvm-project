//===--------------------------- new.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_BUILDING_NEW

#include <stdlib.h>

#include "new"

#if defined(__APPLE__) && !defined(LIBCXXRT)
    #include <cxxabi.h>

    #ifndef _LIBCPPABI_VERSION
        // On Darwin, there are two STL shared libraries and a lower level ABI
        // shared library.  The global holding the current new handler is
        // in the ABI library and named __cxa_new_handler.
        #define __new_handler __cxxabiapple::__cxa_new_handler
    #endif
#else  // __APPLE__
    #if defined(LIBCXXRT) || defined(LIBCXX_BUILDING_LIBCXXABI)
        #include <cxxabi.h>
    #endif  // defined(LIBCXX_BUILDING_LIBCXXABI)
    #if !defined(_LIBCPPABI_VERSION) && !defined(__GLIBCXX__)
        static std::new_handler __new_handler;
    #endif  // _LIBCPPABI_VERSION
#endif

#ifndef __GLIBCXX__

// Implement all new and delete operators as weak definitions
// in this shared library, so that they can be overridden by programs
// that define non-weak copies of the functions.

_LIBCPP_WEAK
void *
operator new(std::size_t size) _THROW_BAD_ALLOC
{
    if (size == 0)
        size = 1;
    void* p;
    while ((p = ::malloc(size)) == 0)
    {
        // If malloc fails and there is a new_handler,
        // call it to try free up memory.
        std::new_handler nh = std::get_new_handler();
        if (nh)
            nh();
        else
#ifndef _LIBCPP_NO_EXCEPTIONS
            throw std::bad_alloc();
#else
            break;
#endif
    }
    return p;
}

_LIBCPP_WEAK
void *
operator new(std::size_t size, std::align_val_t alignment) _THROW_BAD_ALLOC
{
    if (size == 0)
        size = 1;
    if (static_cast<size_t>(alignment) < sizeof(void*))
      alignment = std::align_val_t(sizeof(void*));
    void* p;
    while (::posix_memalign(&p, static_cast<size_t>(alignment), size) != 0)
    {
        // If posix_memalign fails and there is a new_handler,
        // call it to try free up memory.
        std::new_handler nh = std::get_new_handler();
        if (nh)
            nh();
        else {
#ifndef _LIBCPP_NO_EXCEPTIONS
            throw std::bad_alloc();
#else
            p = nullptr; // posix_memalign doesn't initialize 'p' on failure
            break;
#endif
        }
    }
    return p;
}

_LIBCPP_WEAK
void*
operator new(size_t size, const std::nothrow_t&) _NOEXCEPT
{
    void* p = 0;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        p = ::operator new(size);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
    return p;
}

_LIBCPP_WEAK
void*
operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) _NOEXCEPT
{
    void* p = 0;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        p = ::operator new(size, alignment);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
    return p;
}

_LIBCPP_WEAK
void*
operator new[](size_t size) _THROW_BAD_ALLOC
{
    return ::operator new(size);
}

_LIBCPP_WEAK
void*
operator new[](size_t size, std::align_val_t alignment) _THROW_BAD_ALLOC
{
    return ::operator new(size, alignment);
}

_LIBCPP_WEAK
void*
operator new[](size_t size, const std::nothrow_t&) _NOEXCEPT
{
    void* p = 0;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        p = ::operator new[](size);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
    return p;
}

_LIBCPP_WEAK
void*
operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) _NOEXCEPT
{
    void* p = 0;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif  // _LIBCPP_NO_EXCEPTIONS
        p = ::operator new[](size, alignment);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
    }
#endif  // _LIBCPP_NO_EXCEPTIONS
    return p;
}

_LIBCPP_WEAK
void
operator delete(void* ptr) _NOEXCEPT
{
    if (ptr)
        ::free(ptr);
}

_LIBCPP_WEAK
void
operator delete(void* ptr, std::align_val_t) _NOEXCEPT
{
    if (ptr)
        ::free(ptr);
}

_LIBCPP_WEAK
void
operator delete(void* ptr, const std::nothrow_t&) _NOEXCEPT
{
    ::operator delete(ptr);
}

_LIBCPP_WEAK
void
operator delete(void* ptr, std::align_val_t alignment, const std::nothrow_t&) _NOEXCEPT
{
    ::operator delete(ptr, alignment);
}

_LIBCPP_WEAK
void
operator delete(void* ptr, size_t) _NOEXCEPT
{
    ::operator delete(ptr);
}

_LIBCPP_WEAK
void
operator delete(void* ptr, size_t, std::align_val_t alignment) _NOEXCEPT
{
    ::operator delete(ptr, alignment);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr) _NOEXCEPT
{
    ::operator delete(ptr);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr, std::align_val_t alignment) _NOEXCEPT
{
    ::operator delete(ptr, alignment);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr, const std::nothrow_t&) _NOEXCEPT
{
    ::operator delete[](ptr);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr, std::align_val_t alignment, const std::nothrow_t&) _NOEXCEPT
{
    ::operator delete[](ptr, alignment);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr, size_t) _NOEXCEPT
{
    ::operator delete[](ptr);
}

_LIBCPP_WEAK
void
operator delete[] (void* ptr, size_t, std::align_val_t alignment) _NOEXCEPT
{
    ::operator delete[](ptr, alignment);
}

#endif // !__GLIBCXX__

namespace std
{

#ifndef __GLIBCXX__
const nothrow_t nothrow = {};
#endif

#ifndef _LIBCPPABI_VERSION

#ifndef __GLIBCXX__

new_handler
set_new_handler(new_handler handler) _NOEXCEPT
{
    return __sync_lock_test_and_set(&__new_handler, handler);
}

new_handler
get_new_handler() _NOEXCEPT
{
    return __sync_fetch_and_add(&__new_handler, nullptr);
}

#endif // !__GLIBCXX__

#ifndef LIBCXXRT

bad_alloc::bad_alloc() _NOEXCEPT
{
}

#ifndef __GLIBCXX__

bad_alloc::~bad_alloc() _NOEXCEPT
{
}

const char*
bad_alloc::what() const _NOEXCEPT
{
    return "std::bad_alloc";
}

#endif // !__GLIBCXX__

bad_array_new_length::bad_array_new_length() _NOEXCEPT
{
}

#ifndef __GLIBCXX__

bad_array_new_length::~bad_array_new_length() _NOEXCEPT
{
}

const char*
bad_array_new_length::what() const _NOEXCEPT
{
    return "bad_array_new_length";
}

#endif // !__GLIBCXX__

#endif //LIBCXXRT

bad_array_length::bad_array_length() _NOEXCEPT
{
}

#ifndef __GLIBCXX__

bad_array_length::~bad_array_length() _NOEXCEPT
{
}

const char*
bad_array_length::what() const _NOEXCEPT
{
    return "bad_array_length";
}

#endif // !__GLIBCXX__

#endif // _LIBCPPABI_VERSION

#ifndef LIBSTDCXX

void
__throw_bad_alloc()
{
#ifndef _LIBCPP_NO_EXCEPTIONS
    throw bad_alloc();
#else
    _VSTD::abort();
#endif
}

#endif // !LIBSTDCXX

}  // std
