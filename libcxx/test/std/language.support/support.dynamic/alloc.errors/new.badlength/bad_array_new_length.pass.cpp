//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bad_array_new_length

#include <new>
#include <type_traits>
#include <cassert>

int main(int, char**)
{
    static_assert((std::is_base_of<std::bad_alloc, std::bad_array_new_length>::value),
                  "std::is_base_of<std::bad_alloc, std::bad_array_new_length>::value");
    static_assert(std::is_polymorphic<std::bad_array_new_length>::value,
                 "std::is_polymorphic<std::bad_array_new_length>::value");
    std::bad_array_new_length b;
    std::bad_array_new_length b2 = b;
    b2 = b;
    const char* w = b2.what();
    assert(w);

  return 0;
}
