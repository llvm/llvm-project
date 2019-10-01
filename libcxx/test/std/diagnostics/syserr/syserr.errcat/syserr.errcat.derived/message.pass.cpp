//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_category

// virtual string message(int ev) const = 0;

#include <system_error>
#include <cassert>
#include <string>

#include <stdio.h>

#include "test_macros.h"

int main(int, char**)
{
    const std::error_category& e_cat1 = std::generic_category();
    const std::error_category& e_cat2 = std::system_category();
    std::string m1 = e_cat1.message(5);
    std::string m2 = e_cat2.message(5);
    std::string m3 = e_cat2.message(6);
    assert(!m1.empty());
    assert(!m2.empty());
    assert(!m3.empty());
    assert(m1 == m2);
    assert(m1 != m3);

  return 0;
}
