//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class directory_iterator

// typedef ... value_type;
// typedef ... difference_type;
// typedef ... pointer;
// typedef ... reference;
// typedef ... iterator_category

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


int main(int, char**) {
    using namespace fs;
    using D = directory_iterator;
    ASSERT_SAME_TYPE(D::value_type, directory_entry);
    ASSERT_SAME_TYPE(D::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(D::pointer, const directory_entry*);
    ASSERT_SAME_TYPE(D::reference, const directory_entry&);
    ASSERT_SAME_TYPE(D::iterator_category, std::input_iterator_tag);

  return 0;
}
