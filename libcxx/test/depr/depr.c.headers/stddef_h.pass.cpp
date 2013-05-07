//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <stddef.h>

#include <stddef.h>
#include <type_traits>

#ifndef NULL
#error NULL not defined
#endif

#ifndef offsetof
#error offsetof not defined
#endif

int main()
{
    static_assert(sizeof(size_t) == sizeof(void*),
                  "sizeof(size_t) == sizeof(void*)");
    static_assert(std::is_unsigned<size_t>::value,
                  "std::is_unsigned<size_t>::value");
    static_assert(std::is_integral<size_t>::value,
                  "std::is_integral<size_t>::value");
    static_assert(sizeof(ptrdiff_t) == sizeof(void*),
                  "sizeof(ptrdiff_t) == sizeof(void*)");
    static_assert(std::is_signed<ptrdiff_t>::value,
                  "std::is_signed<ptrdiff_t>::value");
    static_assert(std::is_integral<ptrdiff_t>::value,
                  "std::is_integral<ptrdiff_t>::value");
}
