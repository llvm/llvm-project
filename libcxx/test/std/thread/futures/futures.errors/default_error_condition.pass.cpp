//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <future>

// const error_category& future_category();

// virtual error_condition default_error_condition(int ev) const;

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    const std::error_category& e_cat = std::future_category();
    std::error_condition e_cond = e_cat.default_error_condition(static_cast<int>(std::errc::not_a_directory));
    assert(e_cond.category() == e_cat);
    assert(e_cond.value() == static_cast<int>(std::errc::not_a_directory));

  return 0;
}
