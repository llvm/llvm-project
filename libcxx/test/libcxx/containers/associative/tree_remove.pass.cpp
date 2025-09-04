//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Not a portable test

// Returns __tree_next(__z)
// template <class _NodePtr>
// void
// __tree_remove(_NodePtr __root, _NodePtr __z)

#include <__tree>
#include <cassert>

#include "test_macros.h"

struct Node {
  Node* __left_;
  Node* __right_;
  Node* __parent_;
  std::__tree_color __color_;

  Node* __parent_unsafe() const { return __parent_; }
  void __set_parent(Node* x) { __parent_ = x; }
  Node* __get_parent() { return __parent_; }
  void __set_color(std::__tree_color __color) { __color_ = __color; }
  std::__tree_color __get_color() { return __color_; }

  Node() : __left_(), __right_(), __parent_(), __color_() {}
};

void test1() {
  {
    // Left
    // Case 1 -> Case 2 -> x is red turned to black
    Node root;
    Node b;
    Node c;
    Node d;
    Node e;
    Node y;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &y;
    b.__right_    = &d;
    b.__color_ = std::__tree_color::__black;

    y.__parent_   = &b;
    y.__left_     = 0;
    y.__right_    = 0;
    y.__color_ = std::__tree_color::__black;

    d.__parent_   = &b;
    d.__left_     = &c;
    d.__right_    = &e;
    d.__color_ = std::__tree_color::__red;

    c.__parent_   = &d;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    e.__parent_   = &d;
    e.__left_     = 0;
    e.__right_    = 0;
    e.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &y);
    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &d);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(d.__parent_ == &root);
    assert(d.__left_ == &b);
    assert(d.__right_ == &e);
    assert(d.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &d);
    assert(b.__left_ == 0);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(e.__parent_ == &d);
    assert(e.__left_ == 0);
    assert(e.__right_ == 0);
    assert(e.__color_ == std::__tree_color::__black);
  }
  {
    // Right
    // Case 1 -> Case 2 -> x is red turned to black
    Node root;
    Node b;
    Node c;
    Node d;
    Node e;
    Node y;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__right_    = &y;
    b.__left_     = &d;
    b.__color_ = std::__tree_color::__black;

    y.__parent_   = &b;
    y.__right_    = 0;
    y.__left_     = 0;
    y.__color_ = std::__tree_color::__black;

    d.__parent_   = &b;
    d.__right_    = &c;
    d.__left_     = &e;
    d.__color_ = std::__tree_color::__red;

    c.__parent_   = &d;
    c.__right_    = 0;
    c.__left_     = 0;
    c.__color_ = std::__tree_color::__black;

    e.__parent_   = &d;
    e.__right_    = 0;
    e.__left_     = 0;
    e.__color_ = std::__tree_color::__black;

    assert(std::__tree_invariant(root.__left_));
    std::__tree_remove(root.__left_, &y);
    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &d);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(d.__parent_ == &root);
    assert(d.__right_ == &b);
    assert(d.__left_ == &e);
    assert(d.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &d);
    assert(b.__right_ == 0);
    assert(b.__left_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__right_ == 0);
    assert(c.__left_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(e.__parent_ == &d);
    assert(e.__right_ == 0);
    assert(e.__left_ == 0);
    assert(e.__color_ == std::__tree_color::__black);
  }
  {
    // Left
    // Case 1 -> Case 3 -> Case 4
    Node root;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node y;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &y;
    b.__right_    = &d;
    b.__color_ = std::__tree_color::__black;

    y.__parent_   = &b;
    y.__left_     = 0;
    y.__right_    = 0;
    y.__color_ = std::__tree_color::__black;

    d.__parent_   = &b;
    d.__left_     = &c;
    d.__right_    = &e;
    d.__color_ = std::__tree_color::__red;

    c.__parent_   = &d;
    c.__left_     = &f;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    e.__parent_   = &d;
    e.__left_     = 0;
    e.__right_    = 0;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &c;
    f.__left_     = 0;
    f.__right_    = 0;
    f.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &y);
    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &d);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(d.__parent_ == &root);
    assert(d.__left_ == &f);
    assert(d.__right_ == &e);
    assert(d.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &d);
    assert(f.__left_ == &b);
    assert(f.__right_ == &c);
    assert(f.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &f);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &f);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &d);
    assert(e.__left_ == 0);
    assert(e.__right_ == 0);
    assert(e.__color_ == std::__tree_color::__black);
  }
  {
    // Right
    // Case 1 -> Case 3 -> Case 4
    Node root;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node y;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__right_    = &y;
    b.__left_     = &d;
    b.__color_ = std::__tree_color::__black;

    y.__parent_   = &b;
    y.__right_    = 0;
    y.__left_     = 0;
    y.__color_ = std::__tree_color::__black;

    d.__parent_   = &b;
    d.__right_    = &c;
    d.__left_     = &e;
    d.__color_ = std::__tree_color::__red;

    c.__parent_   = &d;
    c.__right_    = &f;
    c.__left_     = 0;
    c.__color_ = std::__tree_color::__black;

    e.__parent_   = &d;
    e.__right_    = 0;
    e.__left_     = 0;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &c;
    f.__right_    = 0;
    f.__left_     = 0;
    f.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &y);
    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &d);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(d.__parent_ == &root);
    assert(d.__right_ == &f);
    assert(d.__left_ == &e);
    assert(d.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &d);
    assert(f.__right_ == &b);
    assert(f.__left_ == &c);
    assert(f.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &f);
    assert(b.__right_ == 0);
    assert(b.__left_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &f);
    assert(c.__right_ == 0);
    assert(c.__left_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &d);
    assert(e.__right_ == 0);
    assert(e.__left_ == 0);
    assert(e.__color_ == std::__tree_color::__black);
  }
}

void test2() {
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &c);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &a);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &c);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &a);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &c);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &a);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &a);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &root);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &c);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &c);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &a);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &a);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &root);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &a);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &root);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &a);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &root);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &b;

    b.__parent_   = &root;
    b.__left_     = &a;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__black;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_remove(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == &b);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    std::__tree_remove(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__parent_ == 0);
    assert(root.__left_ == 0);
    assert(root.__right_ == 0);
    assert(root.__color_ == std::__tree_color::__red);
  }
}

void test3() {
  Node root;
  Node a;
  Node b;
  Node c;
  Node d;
  Node e;
  Node f;
  Node g;
  Node h;

  root.__left_ = &e;

  e.__parent_   = &root;
  e.__left_     = &c;
  e.__right_    = &g;
  e.__color_ = std::__tree_color::__black;

  c.__parent_   = &e;
  c.__left_     = &b;
  c.__right_    = &d;
  c.__color_ = std::__tree_color::__red;

  g.__parent_   = &e;
  g.__left_     = &f;
  g.__right_    = &h;
  g.__color_ = std::__tree_color::__red;

  b.__parent_   = &c;
  b.__left_     = &a;
  b.__right_    = 0;
  b.__color_ = std::__tree_color::__black;

  d.__parent_   = &c;
  d.__left_     = 0;
  d.__right_    = 0;
  d.__color_ = std::__tree_color::__black;

  f.__parent_   = &g;
  f.__left_     = 0;
  f.__right_    = 0;
  f.__color_ = std::__tree_color::__black;

  h.__parent_   = &g;
  h.__left_     = 0;
  h.__right_    = 0;
  h.__color_ = std::__tree_color::__black;

  a.__parent_   = &b;
  a.__left_     = 0;
  a.__right_    = 0;
  a.__color_ = std::__tree_color::__red;

  assert(std::__tree_invariant(root.__left_));

  std::__tree_remove(root.__left_, &h);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &e);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(e.__parent_ == &root);
  assert(e.__left_ == &c);
  assert(e.__right_ == &g);
  assert(e.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &e);
  assert(c.__left_ == &b);
  assert(c.__right_ == &d);
  assert(c.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &e);
  assert(g.__left_ == &f);
  assert(g.__right_ == 0);
  assert(g.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &c);
  assert(b.__left_ == &a);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &c);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &g);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &g);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &e);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(e.__parent_ == &root);
  assert(e.__left_ == &c);
  assert(e.__right_ == &f);
  assert(e.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &e);
  assert(c.__left_ == &b);
  assert(c.__right_ == &d);
  assert(c.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &c);
  assert(b.__left_ == &a);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &c);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &e);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &f);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &c);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &root);
  assert(c.__left_ == &b);
  assert(c.__right_ == &e);
  assert(c.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &c);
  assert(b.__left_ == &a);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &c);
  assert(e.__left_ == &d);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &e);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &e);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &c);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &root);
  assert(c.__left_ == &b);
  assert(c.__right_ == &d);
  assert(c.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &c);
  assert(b.__left_ == &a);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &c);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &d);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == &c);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &b);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &c);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &b);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &a);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(a.__parent_ == &root);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &a);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == 0);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);
}

void test4() {
  Node root;
  Node a;
  Node b;
  Node c;
  Node d;
  Node e;
  Node f;
  Node g;
  Node h;

  root.__left_ = &d;

  d.__parent_   = &root;
  d.__left_     = &b;
  d.__right_    = &f;
  d.__color_ = std::__tree_color::__black;

  b.__parent_   = &d;
  b.__left_     = &a;
  b.__right_    = &c;
  b.__color_ = std::__tree_color::__red;

  f.__parent_   = &d;
  f.__left_     = &e;
  f.__right_    = &g;
  f.__color_ = std::__tree_color::__red;

  a.__parent_   = &b;
  a.__left_     = 0;
  a.__right_    = 0;
  a.__color_ = std::__tree_color::__black;

  c.__parent_   = &b;
  c.__left_     = 0;
  c.__right_    = 0;
  c.__color_ = std::__tree_color::__black;

  e.__parent_   = &f;
  e.__left_     = 0;
  e.__right_    = 0;
  e.__color_ = std::__tree_color::__black;

  g.__parent_   = &f;
  g.__left_     = 0;
  g.__right_    = &h;
  g.__color_ = std::__tree_color::__black;

  h.__parent_   = &g;
  h.__left_     = 0;
  h.__right_    = 0;
  h.__color_ = std::__tree_color::__red;

  assert(std::__tree_invariant(root.__left_));

  std::__tree_remove(root.__left_, &a);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &d);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &root);
  assert(d.__left_ == &b);
  assert(d.__right_ == &f);
  assert(d.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &d);
  assert(b.__left_ == 0);
  assert(b.__right_ == &c);
  assert(b.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &d);
  assert(f.__left_ == &e);
  assert(f.__right_ == &g);
  assert(f.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &b);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__red);

  assert(e.__parent_ == &f);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__black);

  assert(g.__parent_ == &f);
  assert(g.__left_ == 0);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &b);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &d);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &root);
  assert(d.__left_ == &c);
  assert(d.__right_ == &f);
  assert(d.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &d);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &d);
  assert(f.__left_ == &e);
  assert(f.__right_ == &g);
  assert(f.__color_ == std::__tree_color::__red);

  assert(e.__parent_ == &f);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__black);

  assert(g.__parent_ == &f);
  assert(g.__left_ == 0);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &c);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &f);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &root);
  assert(f.__left_ == &d);
  assert(f.__right_ == &g);
  assert(f.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &f);
  assert(d.__left_ == 0);
  assert(d.__right_ == &e);
  assert(d.__color_ == std::__tree_color::__black);

  assert(g.__parent_ == &f);
  assert(g.__left_ == 0);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &d);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &d);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &f);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &root);
  assert(f.__left_ == &e);
  assert(f.__right_ == &g);
  assert(f.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &f);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__black);

  assert(g.__parent_ == &f);
  assert(g.__left_ == 0);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &e);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &g);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &root);
  assert(g.__left_ == &f);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &g);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &f);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &g);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &root);
  assert(g.__left_ == 0);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  std::__tree_remove(root.__left_, &g);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &h);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &root);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  std::__tree_remove(root.__left_, &h);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == 0);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);
}

int main(int, char**) {
  test1();
  test2();
  test3();
  test4();

  return 0;
}
