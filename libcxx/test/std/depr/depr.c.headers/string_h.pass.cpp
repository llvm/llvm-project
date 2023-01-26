//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string.h>

#include <string.h>
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
        size_t s = 0;
        void* vp = 0;
        const void* vpc = 0;
        char* cp = 0;
        const char* cpc = 0;
        ASSERT_SAME_TYPE(void*,         decltype(memcpy(vp, vpc, s)));
        ASSERT_SAME_TYPE(void*,         decltype(memmove(vp, vpc, s)));
        ASSERT_SAME_TYPE(char*,         decltype(strcpy(cp, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(strncpy(cp, cpc, s)));
        ASSERT_SAME_TYPE(char*,         decltype(strcat(cp, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(strncat(cp, cpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(memcmp(vpc, vpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(strcmp(cpc, cpc)));
        ASSERT_SAME_TYPE(int,           decltype(strncmp(cpc, cpc, s)));
        ASSERT_SAME_TYPE(int,           decltype(strcoll(cpc, cpc)));
        ASSERT_SAME_TYPE(size_t,        decltype(strxfrm(cp, cpc, s)));
        ASSERT_SAME_TYPE(size_t,        decltype(strcspn(cpc, cpc)));
        ASSERT_SAME_TYPE(size_t,        decltype(strspn(cpc, cpc)));
        ASSERT_SAME_TYPE(char*,         decltype(strtok(cp, cpc)));
        ASSERT_SAME_TYPE(void*,         decltype(memset(vp, 0, s)));
        ASSERT_SAME_TYPE(char*,         decltype(strerror(0)));
        ASSERT_SAME_TYPE(size_t,        decltype(strlen(cpc)));
    }

    // Functions we (may) reimplement
    {
        // const char* strchr(const char*, int)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(strchr(s, 'l')));
        const char* res = strchr(s, 'l');
        assert(res == &s[2]);
    }
    {
        // char* strchr(char*, int)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(strchr(s, 'l')));
        char* res = strchr(s, 'l');
        assert(res == &s[2]);
    }

    {
        // const char* strpbrk(const char*, const char*)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(strpbrk(s, "el")));
        const char* res = strpbrk(s, "el");
        assert(res == &s[1]);
    }
    {
        // char* strpbrk(char*, const char*)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(strpbrk(s, "el")));
        char* res = strpbrk(s, "el");
        assert(res == &s[1]);
    }

    {
        // const char* strrchr(const char*, int)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(strrchr(s, 'l')));
        const char* res = strrchr(s, 'l');
        assert(res == &s[9]);
    }
    {
        // char* strrchr(char*, int)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(strrchr(s, 'l')));
        char* res = strrchr(s, 'l');
        assert(res == &s[9]);
    }

    {
        // const void* memchr(const void*, int, size_t)
        char storage[] = "hello world";
        size_t count = 11;
        const void* s = storage;
        ASSERT_SAME_TYPE(const void*, decltype(memchr(s, 'l', count)));
        const void* res = memchr(s, 'l', count);
        assert(res == &storage[2]);
    }
    {
        // void* memchr(void*, int, size_t)
        char storage[] = "hello world";
        size_t count = 11;
        void* s = storage;
        ASSERT_SAME_TYPE(void*, decltype(memchr(s, 'l', count)));
        void* res = memchr(s, 'l', count);
        assert(res == &storage[2]);
    }

    {
        // const char* strstr(const char*, const char*)
        char storage[] = "hello world";
        const char* s = storage;
        ASSERT_SAME_TYPE(const char*, decltype(strstr(s, "wor")));
        const char* res = strstr(s, "wor");
        assert(res == &storage[6]);
    }
    {
        // char* strstr(char*, const char*)
        char storage[] = "hello world";
        char* s = storage;
        ASSERT_SAME_TYPE(char*, decltype(strstr(s, "wor")));
        char* res = strstr(s, "wor");
        assert(res == &storage[6]);
    }

    return 0;
}
