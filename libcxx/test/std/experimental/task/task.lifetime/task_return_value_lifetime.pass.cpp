// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <experimental/task>
#include <cassert>
#include <iostream>

#include "../counted.hpp"
#include "../sync_wait.hpp"

DEFINE_COUNTED_VARIABLES();

void test_return_value_lifetime()
{
  counted::reset();

  auto f = [](bool x) -> std::experimental::task<counted>
  {
    if (x) {
      counted c;
      co_return std::move(c);
    }
    co_return {};
  };

  {
    auto t = f(true);

    assert(counted::active_instance_count() == 0);
    assert(counted::copy_constructor_count() == 0);
    assert(counted::move_constructor_count() == 0);

    {
      auto c = sync_wait(std::move(t));
      assert(c.id() == 1);

      assert(counted::active_instance_count() == 2);
      assert(counted::copy_constructor_count() == 0);
      assert(counted::move_constructor_count() > 0);
      assert(counted::default_constructor_count() == 1);
    }

    // The result value in 't' is still alive until 't' destructs.
    assert(counted::active_instance_count() == 1);
  }

  assert(counted::active_instance_count() == 0);

  counted::reset();

  {
    auto t = f(false);

    assert(counted::active_instance_count() == 0);
    assert(counted::copy_constructor_count() == 0);
    assert(counted::move_constructor_count() == 0);

    {
      auto c = sync_wait(std::move(t));
      assert(c.id() == 1);

      assert(counted::active_instance_count() == 2);
      assert(counted::copy_constructor_count() == 0);
      assert(counted::move_constructor_count() > 0);
      assert(counted::default_constructor_count() == 1);
    }

    // The result value in 't' is still alive until 't' destructs.
    assert(counted::active_instance_count() == 1);
  }
}

int main()
{
  test_return_value_lifetime();
  return 0;
}
