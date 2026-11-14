//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>

// template<class Ref, class Val = void, class Allocator = void>
//   class generator;

#include <generator>

#include <cassert>
#include <initializer_list>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

std::generator<int> fib() {
  int a = 0;
  int b = 1;
  while (true) {
    co_yield std::exchange(a, std::exchange(b, a + b));
  }
}

std::generator<int> recursive_fib(int depth) {
  if (depth == 0) {
    co_yield std::ranges::elements_of(fib());
  } else {
    co_yield std::ranges::elements_of(recursive_fib(depth - 1));
  }
};

struct tree_node {
  tree_node* left;
  tree_node* right;
  int element;

  ~tree_node() {
    delete left;
    delete right;
  }
};

tree_node* build_tree(int depth) {
  if (depth == 0) {
    return nullptr;
  }

  tree_node* root = new tree_node();
  root->element   = depth;
  root->left      = build_tree(depth - 1);
  root->right     = build_tree(depth - 1);
  return root;
}

std::generator<int> traversal(tree_node* node) {
  if (node == nullptr) {
    co_return;
  }
  co_yield std::ranges::elements_of(traversal(node->left));
  co_yield node->element;
  co_yield std::ranges::elements_of(traversal(node->right));
}

bool test() {
  {
    std::vector<int> expected_fib_vec = {0, 1, 1, 2, 3};
    {
      auto fib_vec = recursive_fib(1) | std::views::take(5) | std::ranges::to<std::vector<int>>();
      assert(fib_vec == expected_fib_vec);
    }
    {
      auto fib_vec = recursive_fib(42) | std::views::take(5) | std::ranges::to<std::vector<int>>();
      assert(fib_vec == expected_fib_vec);
    }
  }
  {
    tree_node* tree_root = build_tree(10);
    auto node_vec        = traversal(tree_root) | std::ranges::to<std::vector<int>>();
    assert(node_vec.size() == 1023);
    assert(std::reduce(node_vec.cbegin(), node_vec.cend()) == 2036);
    delete tree_root;
  }
  return true;
}

struct Lifetime {
  int& live;
  explicit Lifetime(int& count) : live(count) { ++live; }
  ~Lifetime() { --live; }
};

enum class Failure { None, BeforeYield, AfterYield };

std::generator<int> recursive_lifetime(int depth, int& live, Failure failure) {
  Lifetime lifetime(live);
  if (depth != 0) {
    co_yield std::ranges::elements_of(recursive_lifetime(depth - 1, live, failure));
  } else {
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (failure == Failure::BeforeYield)
      throw 42;
#endif
    co_yield 1;
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (failure == Failure::AfterYield)
      throw 42;
#endif
  }
}

void test_lifetime() {
  int live = 0;
  {
    auto g = recursive_lifetime(8, live, Failure::None);
    assert(live == 0);
  }
  assert(live == 0);
  {
    auto g = recursive_lifetime(8, live, Failure::None);
    auto i = g.begin();
    assert(*i == 1 && live == 9);
    auto moved = std::move(g);
    assert(*i == 1 && live == 9);
    // Destroying the root while the leaf is suspended destroys the entire chain.
  }
  assert(live == 0);
  {
    auto g = recursive_lifetime(8, live, Failure::None);
    auto i = g.begin();
    ++i;
    assert(i == g.end() && live == 0);
  }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
std::generator<int> recover(int& live, Failure failure) {
  try {
    co_yield std::ranges::elements_of(recursive_lifetime(8, live, failure));
    assert(false);
  } catch (int value) {
    assert(value == 42 && live == 0);
  }
  co_yield 2;
}

void test_exceptions() {
  for (Failure failure : {Failure::BeforeYield, Failure::AfterYield}) {
    for (int depth : {0, 8}) {
      int live = 0;
      try {
        auto g = recursive_lifetime(depth, live, failure);
        auto i = g.begin();
        assert(failure == Failure::AfterYield && *i == 1);
        ++i;
        assert(false);
      } catch (int value) {
        assert(value == 42 && live == 0);
      }
    }
    int live = 0;
    auto g   = recover(live, failure);
    auto i   = g.begin();
    if (failure == Failure::AfterYield) {
      assert(*i == 1 && live == 9);
      ++i;
    }
    assert(*i == 2 && live == 0);
    ++i;
    assert(i == g.end());
  }
}
#endif

int main(int, char**) {
  test();
  test_lifetime();
#ifndef TEST_HAS_NO_EXCEPTIONS
  test_exceptions();
#endif
  return 0;
}
