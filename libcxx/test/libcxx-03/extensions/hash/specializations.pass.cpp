//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-modules-build

// Prevent <ext/hash_set> from generating deprecated warnings for this test.
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated
#include <assert.h>
#include <ext/hash_map>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    char str[] = "test";
    assert(__gnu_cxx::hash<const char *>()("test") ==
           std::hash<std::string>()("test"));
    assert(__gnu_cxx::hash<char *>()(str) == std::hash<std::string>()("test"));
    assert(__gnu_cxx::hash<char>()(42) == 42);
    assert(__gnu_cxx::hash<signed char>()(42) == 42);
    assert(__gnu_cxx::hash<unsigned char>()(42) == 42);
    assert(__gnu_cxx::hash<short>()(42) == 42);
    assert(__gnu_cxx::hash<unsigned short>()(42) == 42);
    assert(__gnu_cxx::hash<int>()(42) == 42);
    assert(__gnu_cxx::hash<unsigned int>()(42) == 42);
    assert(__gnu_cxx::hash<long>()(42) == 42);
    assert(__gnu_cxx::hash<unsigned long>()(42) == 42);

  return 0;
}
