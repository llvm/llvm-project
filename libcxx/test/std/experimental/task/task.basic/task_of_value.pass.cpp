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
#include <string>
#include <vector>
#include <memory>
#include "../sync_wait.hpp"

void test_returning_move_only_type()
{
  auto move_only_async =
   [](bool x) -> std::experimental::task<std::unique_ptr<int>> {
     if (x) {
       auto p = std::make_unique<int>(123);
       co_return p; // Should be implicit std::move(p) here.
     }

     co_return std::make_unique<int>(456);
   };

  assert(*sync_wait(move_only_async(true)) == 123);
  assert(*sync_wait(move_only_async(false)) == 456);
}

void test_co_return_with_curly_braces()
{
  auto t = []() -> std::experimental::task<std::tuple<int, std::string>>
  {
    co_return { 123, "test" };
  }();

  auto result = sync_wait(std::move(t));

  assert(std::get<0>(result) == 123);
  assert(std::get<1>(result) == "test");
}

void test_co_return_by_initialiser_list()
{
  auto t = []() -> std::experimental::task<std::vector<int>>
  {
    co_return { 2, 10, -1 };
  }();

  auto result = sync_wait(std::move(t));

  assert(result.size() == 3);
  assert(result[0] == 2);
  assert(result[1] == 10);
  assert(result[2] == -1);
}

int main()
{
  test_returning_move_only_type();
  test_co_return_with_curly_braces();
  test_co_return_by_initialiser_list();

  return 0;
}
