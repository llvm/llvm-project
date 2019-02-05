//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <typeindex>

// class type_index

// type_index(const type_info& rhs);

#include <typeinfo>
#include <typeindex>
#include <cassert>

int main()
{
    std::type_info const & info = typeid(int);
    std::type_index t1(info);
    assert(t1.name() == info.name());
}
