//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_category

// bool operator<(const error_category& rhs) const;

#include <system_error>
#include <cassert>

int main()
{
    const std::error_category& e_cat1 = std::generic_category();
    const std::error_category& e_cat2 = std::generic_category();
    const std::error_category& e_cat3 = std::system_category();
    assert(!(e_cat1 < e_cat2) && !(e_cat2 < e_cat1));
    assert((e_cat1 < e_cat3) || (e_cat3 < e_cat1));
}
