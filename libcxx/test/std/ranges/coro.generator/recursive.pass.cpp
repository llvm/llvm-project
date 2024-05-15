//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::generator

#include <generator>

#include <cassert>
#include <ranges>
#include <utility>
#include <vector>

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
    delete tree_root;
  }
  return true;
}

int main() {
  test();
  return 0;
}
