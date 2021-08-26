//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string.h>

#include <string.h>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

int main(int, char**)
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
    ASSERT_SAME_TYPE(void*,         decltype(memchr(vp, 0, s)));
    ASSERT_SAME_TYPE(const void*,   decltype(memchr(vpc, 0, s)));
    ASSERT_SAME_TYPE(char*,         decltype(strchr(cp, 0)));
    ASSERT_SAME_TYPE(const char*,   decltype(strchr(cpc, 0)));
    ASSERT_SAME_TYPE(size_t,        decltype(strcspn(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,         decltype(strpbrk(cp, cpc)));
    ASSERT_SAME_TYPE(const char*,   decltype(strpbrk(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,         decltype(strrchr(cp, 0)));
    ASSERT_SAME_TYPE(const char*,   decltype(strrchr(cpc, 0)));
    ASSERT_SAME_TYPE(size_t,        decltype(strspn(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,         decltype(strstr(cp, cpc)));
    ASSERT_SAME_TYPE(const char*,   decltype(strstr(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,         decltype(strtok(cp, cpc)));
    ASSERT_SAME_TYPE(void*,         decltype(memset(vp, 0, s)));
    ASSERT_SAME_TYPE(char*,         decltype(strerror(0)));
    ASSERT_SAME_TYPE(size_t,        decltype(strlen(cpc)));

    return 0;
}
