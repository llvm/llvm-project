//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class value_compare

// REQUIRES: c++98 || c++03 || c++11 || c++14

#include <map>
#include <string>

#include "test_macros.h"

int main(int, char**) {
    typedef std::map<int, std::string> map_type;
    typedef map_type::value_compare value_compare;
    typedef map_type::value_type value_type;

    ASSERT_SAME_TYPE(value_compare::result_type, bool);
    ASSERT_SAME_TYPE(value_compare::first_argument_type, value_type);
    ASSERT_SAME_TYPE(value_compare::second_argument_type, value_type);

    return 0;
}
