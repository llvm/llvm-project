//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test type_info

#include <typeinfo>
#include <cstring>
#include <cassert>

int main()
{
    const std::type_info& t1 = typeid(int);
    const std::type_info& t2 = typeid(int);
    const std::type_info& t3 = typeid(short);
    assert(t1.hash_code() == t2.hash_code());
    assert(t1.hash_code() != t3.hash_code());
}
