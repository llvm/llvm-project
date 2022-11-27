//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cstring>

#include <cstring>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

int main(int, char**)
{
    // Functions we get directly from the C library (just check the signature)
    {
        std::size_t s = 0;
        void* vp = 0;
        const void* vpc = 0;
        char* cp = 0;
        const char* cpc = 0;
        ASSERT_SAME_TYPE(void*,         decltype(std::memcpy(vp, vpc, s)));
        ASSERT_SAME_TYPE(void*,         decltype(std::memmove(vp, vpc, s)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strcpy(cp, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strncpy(cp, cpc, s)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strcat(cp, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strncat(cp, cpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(std::memcmp(vpc, vpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(std::strcmp(cpc, cpc)));
        ASSERT_SAME_TYPE(int,           decltype(std::strncmp(cpc, cpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(std::strcoll(cpc, cpc)));
        ASSERT_SAME_TYPE(std::size_t,   decltype(std::strxfrm(cp, cpc, s)));
        ASSERT_SAME_TYPE(std::size_t,   decltype(std::strcspn(cpc, cpc)));
        ASSERT_SAME_TYPE(std::size_t,   decltype(std::strspn(cpc, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strtok(cp, cpc)));
        ASSERT_SAME_TYPE(void*,         decltype(std::memset(vp, 0, s)));
        ASSERT_SAME_TYPE(char*,         decltype(std::strerror(0)));
        ASSERT_SAME_TYPE(std::size_t,   decltype(std::strlen(cpc)));
    }

    // Functions we (may) reimplement
    {
        // const char* strchr(const char*, int)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(std::strchr(s, 'l')));
        const char* res = std::strchr(s, 'l');
        assert(res == &s[2]);
    }
    {
        // char* strchr(char*, int)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(std::strchr(s, 'l')));
        char* res = std::strchr(s, 'l');
        assert(res == &s[2]);
    }

    {
        // const char* strpbrk(const char*, const char*)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(std::strpbrk(s, "el")));
        const char* res = std::strpbrk(s, "el");
        assert(res == &s[1]);
    }
    {
        // char* strpbrk(char*, const char*)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(std::strpbrk(s, "el")));
        char* res = std::strpbrk(s, "el");
        assert(res == &s[1]);
    }

    {
        // const char* strrchr(const char*, int)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(std::strrchr(s, 'l')));
        const char* res = std::strrchr(s, 'l');
        assert(res == &s[9]);
    }
    {
        // char* strrchr(char*, int)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(std::strrchr(s, 'l')));
        char* res = std::strrchr(s, 'l');
        assert(res == &s[9]);
    }

    {
        // const void* memchr(const void*, int, size_t)
        char storage[] = "hello world";
        std::size_t count = 11;
        const void* s = storage;
        ASSERT_SAME_TYPE(const void*, decltype(std::memchr(s, 'l', count)));
        const void* res = std::memchr(s, 'l', count);
        assert(res == &storage[2]);
    }
    {
        // void* memchr(void*, int, size_t)
        char storage[] = "hello world";
        std::size_t count = 11;
        void* s = storage;
        ASSERT_SAME_TYPE(void*, decltype(std::memchr(s, 'l', count)));
        void* res = std::memchr(s, 'l', count);
        assert(res == &storage[2]);
    }

    {
        // const char* strstr(const char*, const char*)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(std::strstr(s, "wor")));
        const char* res = std::strstr(s, "wor");
        assert(res == &storage[6]);
    }
    {
        // char* strstr(char*, const char*)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(std::strstr(s, "wor")));
        char* res = std::strstr(s, "wor");
        assert(res == &storage[6]);
    }

    return 0;
}
