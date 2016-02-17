//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cstring>

#include <cstring>
#include <type_traits>

#ifndef NULL
#error NULL not defined
#endif

int main()
{
    std::size_t s = 0;
    void* vp = 0;
    const void* vpc = 0;
    char* cp = 0;
    const char* cpc = 0;
    static_assert((std::is_same<decltype(std::memcpy(vp, vpc, s)), void*>::value), "");
    static_assert((std::is_same<decltype(std::memmove(vp, vpc, s)), void*>::value), "");
    static_assert((std::is_same<decltype(std::strcpy(cp, cpc)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strncpy(cp, cpc, s)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strcat(cp, cpc)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strncat(cp, cpc, s)), char*>::value), "");
    static_assert((std::is_same<decltype(std::memcmp(vpc, vpc, s)), int>::value), "");
    static_assert((std::is_same<decltype(std::strcmp(cpc, cpc)), int>::value), "");
    static_assert((std::is_same<decltype(std::strncmp(cpc, cpc, s)), int>::value), "");
    static_assert((std::is_same<decltype(std::strcoll(cpc, cpc)), int>::value), "");
    static_assert((std::is_same<decltype(std::strxfrm(cp, cpc, s)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::memchr(vp, 0, s)), void*>::value), "");
    static_assert((std::is_same<decltype(std::strchr(cp, 0)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strcspn(cpc, cpc)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::strpbrk(cp, cpc)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strrchr(cp, 0)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strspn(cpc, cpc)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::strstr(cp, cpc)), char*>::value), "");
#ifndef _LIBCPP_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS
    static_assert((std::is_same<decltype(std::strtok(cp, cpc)), char*>::value), "");
#endif
    static_assert((std::is_same<decltype(std::memset(vp, 0, s)), void*>::value), "");
    static_assert((std::is_same<decltype(std::strerror(0)), char*>::value), "");
    static_assert((std::is_same<decltype(std::strlen(cpc)), std::size_t>::value), "");

    // These tests fail on systems whose C library doesn't provide a correct overload
    // set for strchr, strpbrk, strrchr, strstr, and memchr, unless the compiler is
    // a suitably recent version of Clang.
#if !defined(__APPLE__) || defined(_LIBCPP_PREFERRED_OVERLOAD)
    static_assert((std::is_same<decltype(std::memchr(vpc, 0, s)), const void*>::value), "");
    static_assert((std::is_same<decltype(std::strchr(cpc, 0)), const char*>::value), "");
    static_assert((std::is_same<decltype(std::strpbrk(cpc, cpc)), const char*>::value), "");
    static_assert((std::is_same<decltype(std::strrchr(cpc, 0)), const char*>::value), "");
    static_assert((std::is_same<decltype(std::strstr(cpc, cpc)), const char*>::value), "");
#endif
}
