//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// const error_category& future_category();

// virtual bool equivalent(const error_code& code, int condition) const;

#include <future>
#include <cassert>

int main(int, char**)
{
    const std::error_category& e_cat = std::future_category();
    assert(e_cat.equivalent(std::error_code(5, e_cat), 5));
    assert(!e_cat.equivalent(std::error_code(5, e_cat), 6));

  return 0;
}
