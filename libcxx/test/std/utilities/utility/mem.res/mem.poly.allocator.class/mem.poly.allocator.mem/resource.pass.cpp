//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{11.0|12.0}}

// <memory_resource>

// template <class T> class polymorphic_allocator

// memory_resource *
// polymorphic_allocator<T>::resource() const

#include <memory_resource>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  typedef std::pmr::polymorphic_allocator<void> A;
  {
    A const a;
    ASSERT_SAME_TYPE(decltype(a.resource()), std::pmr::memory_resource*);
  }
  {
    std::pmr::memory_resource* mptr = (std::pmr::memory_resource*)42;
    A const a(mptr);
    assert(a.resource() == mptr);
  }
  {
    A const a(nullptr);
    assert(a.resource() == nullptr);
    assert(a.resource() == nullptr);
  }
  {
    A const a;
    assert(a.resource() == std::pmr::get_default_resource());
  }
  {
    std::pmr::memory_resource* mptr = (std::pmr::memory_resource*)42;
    std::pmr::set_default_resource(mptr);
    A const a;
    assert(a.resource() == mptr);
    assert(a.resource() == std::pmr::get_default_resource());
  }

  return 0;
}
