//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__stop_token/intrusive_list_view.h>
#include <cassert>

#include "test_macros.h"

struct Node : std::__intrusive_node_base<Node> {
  int i;

  Node(int ii) : i(ii) {}
};

using ListView = std::__intrusive_list_view<Node>;

int main(int, char**) {
  // empty
  {
    ListView list;
    assert(list.__empty());
  }

  // push_front
  {
    ListView list;
    Node n1(5);
    list.__push_front(&n1);
    assert(!list.__empty());
  }

  // pop_front
  {
    ListView list;
    Node n1(5);
    Node n2(6);
    list.__push_front(&n1);
    list.__push_front(&n2);

    auto f1 = list.__pop_front();
    assert(f1->i == 6);

    auto f2 = list.__pop_front();
    assert(f2->i == 5);

    assert(list.__empty());
  }

  // remove head
  {
    ListView list;
    Node n1(5);
    Node n2(6);
    list.__push_front(&n1);
    list.__push_front(&n2);

    list.__remove(&n2);

    auto f = list.__pop_front();
    assert(f->i == 5);

    assert(list.__empty());
  }

  // remove non-head
  {
    ListView list;
    Node n1(5);
    Node n2(6);
    Node n3(7);
    list.__push_front(&n1);
    list.__push_front(&n2);
    list.__push_front(&n3);

    list.__remove(&n2);

    auto f1 = list.__pop_front();
    assert(f1->i == 7);

    auto f2 = list.__pop_front();
    assert(f2->i == 5);

    assert(list.__empty());
  }

  // is_head
  {
    ListView list;
    Node n1(5);
    Node n2(6);
    list.__push_front(&n1);
    list.__push_front(&n2);

    assert(!list.__is_head(&n1));
    assert(list.__is_head(&n2));
  }

  return 0;
}
