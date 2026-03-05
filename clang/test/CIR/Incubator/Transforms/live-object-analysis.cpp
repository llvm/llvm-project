// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | cir-opt  --pass-pipeline='builtin.module(cir.func(cir-live-object-diagnostics))' --verify-diagnostics -o /dev/null

struct Node {
  Node() = default;
  Node(Node &&) = default;
  // expected-remark@above {{last use of this}}
  Node(const Node &) = default;
  // expected-remark@above {{last use of this}}
  Node &operator=(Node &&) = default;
  // expected-remark@above {{last use of this}}
  // expected-remark@above {{last use of __retval}}
  Node &operator=(const Node &) = default;
  // expected-remark@above {{last use of this}}
  // expected-remark@above {{last use of __retval}}

  int val;
};

int test_copy_ctor() {
  Node orig;
  Node copy(orig);
  // expected-remark@above {{last use of orig}}

  return copy.val;
  // expected-remark@above {{last use of copy}}
  // expected-remark@above {{last use of __retval}}
}

int test_move_ctor() {
  Node orig;
  Node move((Node &&)orig);
  // expected-remark@above {{last use of orig}}

  return move.val;
  // expected-remark@above {{last use of move}}
  // expected-remark@above {{last use of __retval}}
}

int test_copy_assign() {
  Node orig, copy;
  copy = orig;
  // expected-remark@above {{last use of orig}}

  return copy.val;
  // expected-remark@above {{last use of copy}}
  // expected-remark@above {{last use of __retval}}
}

int test_move_assign() {
  Node orig, move;
  move = (Node &&)orig;
  // expected-remark@above {{last use of orig}}

  return move.val;
  // expected-remark@above {{last use of move}}
  // expected-remark@above {{last use of __retval}}
}

int test_move_chain() {
  Node first, second, third, fourth;

  second = first;
  // expected-remark@above {{last use of first}}
  third = second;
  // expected-remark@above {{last use of second}}
  fourth = third;
  // expected-remark@above {{last use of third}}

  return fourth.val;
  // expected-remark@above {{last use of fourth}}
  // expected-remark@above {{last use of __retval}}
}
