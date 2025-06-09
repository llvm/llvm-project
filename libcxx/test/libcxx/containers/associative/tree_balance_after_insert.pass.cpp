//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Not a portable test

// Precondition:  __root->__color_ == std::__tree_color::__black
// template <class _NodePtr>
// void
// __tree_balance_after_insert(_NodePtr __root, _NodePtr __x)

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
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &a;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = 0;
    d.__right_    = 0;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == &a);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == 0);
    assert(d.__right_ == 0);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = 0;
    b.__right_    = &a;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = 0;
    d.__right_    = 0;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == 0);
    assert(b.__right_ == &a);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == 0);
    assert(d.__right_ == 0);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = 0;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = &a;
    d.__right_    = 0;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &d;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == &a);
    assert(d.__right_ == 0);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &d);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = 0;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = 0;
    d.__right_    = &a;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &d;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == 0);
    assert(b.__right_ == 0);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == 0);
    assert(d.__right_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &d);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;
    Node h;
    Node i;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &a;
    b.__right_    = &g;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = &h;
    d.__right_    = &i;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = &e;
    a.__right_    = &f;
    a.__color_ = std::__tree_color::__red;

    e.__parent_   = &a;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &a;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &b;
    g.__color_ = std::__tree_color::__black;

    h.__parent_   = &d;
    h.__color_ = std::__tree_color::__black;

    i.__parent_   = &d;
    i.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == &a);
    assert(b.__right_ == &g);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == &h);
    assert(d.__right_ == &i);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &e);
    assert(a.__right_ == &f);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;
    Node h;
    Node i;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &g;
    b.__right_    = &a;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = &h;
    d.__right_    = &i;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = &e;
    a.__right_    = &f;
    a.__color_ = std::__tree_color::__red;

    e.__parent_   = &a;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &a;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &b;
    g.__color_ = std::__tree_color::__black;

    h.__parent_   = &d;
    h.__color_ = std::__tree_color::__black;

    i.__parent_   = &d;
    i.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == &g);
    assert(b.__right_ == &a);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == &h);
    assert(d.__right_ == &i);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &e);
    assert(a.__right_ == &f);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;
    Node h;
    Node i;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &g;
    b.__right_    = &h;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = &a;
    d.__right_    = &i;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &d;
    a.__left_     = &e;
    a.__right_    = &f;
    a.__color_ = std::__tree_color::__red;

    e.__parent_   = &a;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &a;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &b;
    g.__color_ = std::__tree_color::__black;

    h.__parent_   = &b;
    h.__color_ = std::__tree_color::__black;

    i.__parent_   = &d;
    i.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == &g);
    assert(b.__right_ == &h);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == &a);
    assert(d.__right_ == &i);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &d);
    assert(a.__left_ == &e);
    assert(a.__right_ == &f);
    assert(a.__color_ == std::__tree_color::__red);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;
    Node h;
    Node i;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &d;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &g;
    b.__right_    = &h;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &c;
    d.__left_     = &i;
    d.__right_    = &a;
    d.__color_ = std::__tree_color::__red;

    a.__parent_   = &d;
    a.__left_     = &e;
    a.__right_    = &f;
    a.__color_ = std::__tree_color::__red;

    e.__parent_   = &a;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &a;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &b;
    g.__color_ = std::__tree_color::__black;

    h.__parent_   = &b;
    h.__color_ = std::__tree_color::__black;

    i.__parent_   = &d;
    i.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &c);

    assert(c.__parent_ == &root);
    assert(c.__left_ == &b);
    assert(c.__right_ == &d);
    assert(c.__color_ == std::__tree_color::__black);

    assert(b.__parent_ == &c);
    assert(b.__left_ == &g);
    assert(b.__right_ == &h);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &c);
    assert(d.__left_ == &i);
    assert(d.__right_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(a.__parent_ == &d);
    assert(a.__left_ == &e);
    assert(a.__right_ == &f);
    assert(a.__color_ == std::__tree_color::__red);
  }
}

void test2() {
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &a;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    a.__parent_   = &c;
    a.__left_     = 0;
    a.__right_    = &b;
    a.__color_ = std::__tree_color::__red;

    b.__parent_   = &a;
    b.__left_     = 0;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &a;

    a.__parent_   = &root;
    a.__left_     = 0;
    a.__right_    = &c;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &a;
    c.__left_     = &b;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    b.__parent_   = &c;
    b.__left_     = 0;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &a;
    c.__right_    = &g;
    c.__color_ = std::__tree_color::__black;

    a.__parent_   = &c;
    a.__left_     = &d;
    a.__right_    = &b;
    a.__color_ = std::__tree_color::__red;

    b.__parent_   = &a;
    b.__left_     = &e;
    b.__right_    = &f;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &a;
    d.__color_ = std::__tree_color::__black;

    e.__parent_   = &b;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &b;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &c;
    g.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == &f);
    assert(c.__right_ == &g);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &d);
    assert(a.__right_ == &e);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &a);
    assert(e.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &c);
    assert(f.__color_ == std::__tree_color::__black);

    assert(g.__parent_ == &c);
    assert(g.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;

    root.__left_ = &a;

    a.__parent_   = &root;
    a.__left_     = &d;
    a.__right_    = &c;
    a.__color_ = std::__tree_color::__black;

    c.__parent_   = &a;
    c.__left_     = &b;
    c.__right_    = &g;
    c.__color_ = std::__tree_color::__red;

    b.__parent_   = &c;
    b.__left_     = &e;
    b.__right_    = &f;
    b.__color_ = std::__tree_color::__red;

    d.__parent_   = &a;
    d.__color_ = std::__tree_color::__black;

    e.__parent_   = &b;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &b;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &c;
    g.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &b);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == &f);
    assert(c.__right_ == &g);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &d);
    assert(a.__right_ == &e);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &a);
    assert(e.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &c);
    assert(f.__color_ == std::__tree_color::__black);

    assert(g.__parent_ == &c);
    assert(g.__color_ == std::__tree_color::__black);
  }
}

void test3() {
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &a;
    b.__right_    = 0;
    b.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = 0;
    a.__right_    = 0;
    a.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;

    root.__left_ = &a;

    a.__parent_   = &root;
    a.__left_     = 0;
    a.__right_    = &b;
    a.__color_ = std::__tree_color::__black;

    b.__parent_   = &a;
    b.__left_     = 0;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = 0;
    c.__right_    = 0;
    c.__color_ = std::__tree_color::__red;

    std::__tree_balance_after_insert(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(a.__parent_ == &b);
    assert(a.__left_ == 0);
    assert(a.__right_ == 0);
    assert(a.__color_ == std::__tree_color::__red);

    assert(c.__parent_ == &b);
    assert(c.__left_ == 0);
    assert(c.__right_ == 0);
    assert(c.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;

    root.__left_ = &c;

    c.__parent_   = &root;
    c.__left_     = &b;
    c.__right_    = &g;
    c.__color_ = std::__tree_color::__black;

    b.__parent_   = &c;
    b.__left_     = &a;
    b.__right_    = &f;
    b.__color_ = std::__tree_color::__red;

    a.__parent_   = &b;
    a.__left_     = &d;
    a.__right_    = &e;
    a.__color_ = std::__tree_color::__red;

    d.__parent_   = &a;
    d.__color_ = std::__tree_color::__black;

    e.__parent_   = &a;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &b;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &c;
    g.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &a);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == &f);
    assert(c.__right_ == &g);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &d);
    assert(a.__right_ == &e);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &a);
    assert(e.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &c);
    assert(f.__color_ == std::__tree_color::__black);

    assert(g.__parent_ == &c);
    assert(g.__color_ == std::__tree_color::__black);
  }
  {
    Node root;
    Node a;
    Node b;
    Node c;
    Node d;
    Node e;
    Node f;
    Node g;

    root.__left_ = &a;

    a.__parent_   = &root;
    a.__left_     = &d;
    a.__right_    = &b;
    a.__color_ = std::__tree_color::__black;

    b.__parent_   = &a;
    b.__left_     = &e;
    b.__right_    = &c;
    b.__color_ = std::__tree_color::__red;

    c.__parent_   = &b;
    c.__left_     = &f;
    c.__right_    = &g;
    c.__color_ = std::__tree_color::__red;

    d.__parent_   = &a;
    d.__color_ = std::__tree_color::__black;

    e.__parent_   = &b;
    e.__color_ = std::__tree_color::__black;

    f.__parent_   = &c;
    f.__color_ = std::__tree_color::__black;

    g.__parent_   = &c;
    g.__color_ = std::__tree_color::__black;

    std::__tree_balance_after_insert(root.__left_, &c);

    assert(std::__tree_invariant(root.__left_));

    assert(root.__left_ == &b);

    assert(c.__parent_ == &b);
    assert(c.__left_ == &f);
    assert(c.__right_ == &g);
    assert(c.__color_ == std::__tree_color::__red);

    assert(a.__parent_ == &b);
    assert(a.__left_ == &d);
    assert(a.__right_ == &e);
    assert(a.__color_ == std::__tree_color::__red);

    assert(b.__parent_ == &root);
    assert(b.__left_ == &a);
    assert(b.__right_ == &c);
    assert(b.__color_ == std::__tree_color::__black);

    assert(d.__parent_ == &a);
    assert(d.__color_ == std::__tree_color::__black);

    assert(e.__parent_ == &a);
    assert(e.__color_ == std::__tree_color::__black);

    assert(f.__parent_ == &c);
    assert(f.__color_ == std::__tree_color::__black);

    assert(g.__parent_ == &c);
    assert(g.__color_ == std::__tree_color::__black);
  }
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

  root.__left_ = &a;
  a.__parent_  = &root;

  std::__tree_balance_after_insert(root.__left_, &a);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &a);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(a.__parent_ == &root);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  a.__right_  = &b;
  b.__parent_ = &a;

  std::__tree_balance_after_insert(root.__left_, &b);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &a);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(a.__parent_ == &root);
  assert(a.__left_ == 0);
  assert(a.__right_ == &b);
  assert(a.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &a);
  assert(b.__left_ == 0);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__red);

  b.__right_  = &c;
  c.__parent_ = &b;

  std::__tree_balance_after_insert(root.__left_, &c);

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
  assert(b.__right_ == &c);
  assert(b.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &b);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__red);

  c.__right_  = &d;
  d.__parent_ = &c;

  std::__tree_balance_after_insert(root.__left_, &d);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == &c);
  assert(b.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &b);
  assert(c.__left_ == 0);
  assert(c.__right_ == &d);
  assert(c.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &c);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__red);

  d.__right_  = &e;
  e.__parent_ = &d;

  std::__tree_balance_after_insert(root.__left_, &e);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == &d);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &b);
  assert(d.__left_ == &c);
  assert(d.__right_ == &e);
  assert(d.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &d);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__red);

  assert(e.__parent_ == &d);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__red);

  e.__right_  = &f;
  f.__parent_ = &e;

  std::__tree_balance_after_insert(root.__left_, &f);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == &d);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &b);
  assert(d.__left_ == &c);
  assert(d.__right_ == &e);
  assert(d.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &d);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &d);
  assert(e.__left_ == 0);
  assert(e.__right_ == &f);
  assert(e.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &e);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__red);

  f.__right_  = &g;
  g.__parent_ = &f;

  std::__tree_balance_after_insert(root.__left_, &g);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &b);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(b.__parent_ == &root);
  assert(b.__left_ == &a);
  assert(b.__right_ == &d);
  assert(b.__color_ == std::__tree_color::__black);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &b);
  assert(d.__left_ == &c);
  assert(d.__right_ == &f);
  assert(d.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &d);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__black);

  assert(f.__parent_ == &d);
  assert(f.__left_ == &e);
  assert(f.__right_ == &g);
  assert(f.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &f);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &f);
  assert(g.__left_ == 0);
  assert(g.__right_ == 0);
  assert(g.__color_ == std::__tree_color::__red);

  g.__right_  = &h;
  h.__parent_ = &g;

  std::__tree_balance_after_insert(root.__left_, &h);

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
  assert(b.__left_ == &a);
  assert(b.__right_ == &c);
  assert(b.__color_ == std::__tree_color::__red);

  assert(a.__parent_ == &b);
  assert(a.__left_ == 0);
  assert(a.__right_ == 0);
  assert(a.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &b);
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
}

void test5() {
  Node root;
  Node a;
  Node b;
  Node c;
  Node d;
  Node e;
  Node f;
  Node g;
  Node h;

  root.__left_ = &h;
  h.__parent_  = &root;

  std::__tree_balance_after_insert(root.__left_, &h);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &h);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &root);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  h.__left_   = &g;
  g.__parent_ = &h;

  std::__tree_balance_after_insert(root.__left_, &g);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &h);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &root);
  assert(h.__left_ == &g);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  assert(g.__parent_ == &h);
  assert(g.__left_ == 0);
  assert(g.__right_ == 0);
  assert(g.__color_ == std::__tree_color::__red);

  g.__left_   = &f;
  f.__parent_ = &g;

  std::__tree_balance_after_insert(root.__left_, &f);

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
  assert(f.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__red);

  f.__left_   = &e;
  e.__parent_ = &f;

  std::__tree_balance_after_insert(root.__left_, &e);

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
  assert(f.__left_ == &e);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &f);
  assert(e.__left_ == 0);
  assert(e.__right_ == 0);
  assert(e.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  e.__left_   = &d;
  d.__parent_ = &e;

  std::__tree_balance_after_insert(root.__left_, &d);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &g);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &root);
  assert(g.__left_ == &e);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &g);
  assert(e.__left_ == &d);
  assert(e.__right_ == &f);
  assert(e.__color_ == std::__tree_color::__black);

  assert(d.__parent_ == &e);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &e);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__red);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  d.__left_   = &c;
  c.__parent_ = &d;

  std::__tree_balance_after_insert(root.__left_, &c);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &g);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &root);
  assert(g.__left_ == &e);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &g);
  assert(e.__left_ == &d);
  assert(e.__right_ == &f);
  assert(e.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &e);
  assert(d.__left_ == &c);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__black);

  assert(c.__parent_ == &d);
  assert(c.__left_ == 0);
  assert(c.__right_ == 0);
  assert(c.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &e);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  c.__left_   = &b;
  b.__parent_ = &c;

  std::__tree_balance_after_insert(root.__left_, &b);

  assert(std::__tree_invariant(root.__left_));

  assert(root.__parent_ == 0);
  assert(root.__left_ == &g);
  assert(root.__right_ == 0);
  assert(root.__color_ == std::__tree_color::__red);

  assert(g.__parent_ == &root);
  assert(g.__left_ == &e);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__black);

  assert(e.__parent_ == &g);
  assert(e.__left_ == &c);
  assert(e.__right_ == &f);
  assert(e.__color_ == std::__tree_color::__red);

  assert(c.__parent_ == &e);
  assert(c.__left_ == &b);
  assert(c.__right_ == &d);
  assert(c.__color_ == std::__tree_color::__black);

  assert(b.__parent_ == &c);
  assert(b.__left_ == 0);
  assert(b.__right_ == 0);
  assert(b.__color_ == std::__tree_color::__red);

  assert(d.__parent_ == &c);
  assert(d.__left_ == 0);
  assert(d.__right_ == 0);
  assert(d.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &e);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);

  b.__left_   = &a;
  a.__parent_ = &b;

  std::__tree_balance_after_insert(root.__left_, &a);

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

  assert(g.__parent_ == &e);
  assert(g.__left_ == &f);
  assert(g.__right_ == &h);
  assert(g.__color_ == std::__tree_color::__red);

  assert(f.__parent_ == &g);
  assert(f.__left_ == 0);
  assert(f.__right_ == 0);
  assert(f.__color_ == std::__tree_color::__black);

  assert(h.__parent_ == &g);
  assert(h.__left_ == 0);
  assert(h.__right_ == 0);
  assert(h.__color_ == std::__tree_color::__black);
}

int main(int, char**) {
  test1();
  test2();
  test3();
  test4();
  test5();

  return 0;
}
