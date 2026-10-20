// An assert with a message, assert(p && "why"), proves p non-null after it.
// The macros below are the expansions used by real C libraries, written out so
// the test does not depend on the host's headers.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 -Wno-unused-value %s -verify

[[noreturn]] void __assert_fail(const char *, const char *, unsigned, const char *);
[[noreturn]] void abort_impl();

#define GLIBCXX_ASSERT(e) \
  (static_cast<bool>(e) ? void(0) : __assert_fail(#e, __FILE__, __LINE__, __func__))
#define GLIBC_C_ASSERT(e) ((e) ? (void)0 : __assert_fail(#e, __FILE__, __LINE__, __func__))
#define BSD_ASSERT(e) ((void)((e) ? 0 : (abort_impl(), 0)))

struct Node {
  int value;
  Node *next;
};

Node *_Nullable lookup(int key);

int glibcxx_with_message(int key) {
  Node *n = lookup(key);
  GLIBCXX_ASSERT(n && "key must exist");
  return n->value;
}

int glibcxx_plain(int key) {
  Node *n = lookup(key);
  GLIBCXX_ASSERT(n);
  return n->value;
}

int glibc_c_with_message(int key) {
  Node *n = lookup(key);
  GLIBC_C_ASSERT(n && "key must exist");
  return n->value;
}

int bsd_with_message(int key) {
  Node *n = lookup(key);
  BSD_ASSERT(n && "key must exist");
  return n->value;
}

int two_pointers(int a, int b) {
  Node *x = lookup(a);
  Node *y = lookup(b);
  GLIBCXX_ASSERT(x && y && "both keys must exist");
  return x->value + y->value;
}

int nested_member(Node *_Nullable n) {
  GLIBCXX_ASSERT(n && n->next && "need two nodes");
  return n->next->value;
}

int ternary_whole_and(Node *_Nullable a, Node *_Nullable b) {
  return (a && b) ? a->value + b->value : 0;
}

int while_whole_and(Node *_Nullable a, Node *_Nullable b) {
  int total = 0;
  while (static_cast<bool>(a && b)) {
    total += a->value + b->value;
    a = lookup(total);
  }
  return total;
}

int or_false_edge(Node *_Nullable a, Node *_Nullable b) {
  return (!a || !b) ? 0 : a->value + b->value;
}

int message_does_not_cover_other_pointer(int key, Node *_Nullable other) {
  Node *n = lookup(key);
  GLIBCXX_ASSERT(n && "key must exist");
  return n->value + other->value; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int or_true_edge_proves_nothing(Node *_Nullable a, Node *_Nullable b) {
  return (a || b) ? a->value : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int and_false_edge_proves_nothing(Node *_Nullable a, Node *_Nullable b) {
  return (a && b) ? 0 : a->value; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
