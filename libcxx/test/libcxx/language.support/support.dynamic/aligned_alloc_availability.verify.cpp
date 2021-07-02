//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure we get compile-time availability errors when trying to use aligned
// allocation/deallocation on deployment targets that don't support it.

// UNSUPPORTED: c++03, c++11, c++14

// Aligned allocation was not provided before macosx10.14.
// Support for that is broken prior to Clang 8 and Apple Clang 11.
// UNSUPPORTED: apple-clang-9, apple-clang-10
// UNSUPPORTED: clang-5, clang-6, clang-7

// REQUIRES: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13}}

#include <new>
#include <cstddef>

#include "test_macros.h"

constexpr auto OverAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;

struct alignas(OverAligned) A { };

int main(int, char**)
{
    // Normal versions
    {
        A *a1 = new A; // expected-error-re {{aligned allocation function of type {{.+}} is only available on}}
        // `delete` is also required by the line above if construction fails
        // expected-error-re@-2 {{aligned deallocation function of type {{.+}} is only available on}}

        delete a1; // expected-error-re {{aligned deallocation function of type {{.+}} is only available on}}

        A* a2 = new(std::nothrow) A; // expected-error-re {{aligned allocation function of type {{.+}} is only available on}}
        // `delete` is also required above for the same reason
        // expected-error-re@-2 {{aligned deallocation function of type {{.+}} is only available on}}
    }

    // Array versions
    {
        A *a1 = new A[2]; // expected-error-re {{aligned allocation function of type {{.+}} is only available on}}
        // `delete` is also required by the line above if construction fails
        // expected-error-re@-2 {{aligned deallocation function of type {{.+}} is only available on}}

        delete[] a1; // expected-error-re {{aligned deallocation function of type {{.+}} is only available on}}

        A* a2 = new(std::nothrow) A[2]; // expected-error-re {{aligned allocation function of type {{.+}} is only available on}}
        // `delete` is also required above for the same reason
        // expected-error-re@-2 {{aligned deallocation function of type {{.+}} is only available on}}
    }
}
