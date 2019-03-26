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

#include "../counted.hpp"
#include "../sync_wait.hpp"

DEFINE_COUNTED_VARIABLES();

void test_parameter_lifetime()
{
  counted::reset();

  auto f = [](counted c) -> std::experimental::task<std::size_t>
  {
    co_return c.id();
  };

  {
    auto t = f({});

    assert(counted::active_instance_count() == 1);
    assert(counted::copy_constructor_count() == 0);
    assert(counted::move_constructor_count() <= 2); // Ideally <= 1

    auto id = sync_wait(t);
    assert(id == 1);

    assert(counted::active_instance_count() == 1);
    assert(counted::copy_constructor_count() == 0);

    // We are relying on C++17 copy-elision when passing the temporary counter
    // into f(). Then f() must move the parameter into the coroutine frame by
    // calling the move-constructor. This move could also potentially be
    // elided by the
    assert(counted::move_constructor_count() <= 1);
  }

  assert(counted::active_instance_count() == 0);
}

int main()
{
  test_parameter_lifetime();
  return 0;
}
