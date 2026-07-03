// RUN: %check_clang_tidy -std=c++20-or-later %s misc-use-braced-initialization %t -- --fix-notes

#include <vector>

struct Agg {
  int a, b;
};

struct AggDefault {
  int a = 0;
  int b;
};

struct Nested {
  Agg x;
  int y;
};

struct Takes {
  Takes(Agg);
};

struct Simple {
  Simple(int);
};

void basic_aggregate() {
  Agg d(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization instead of parenthesized initialization [misc-use-braced-initialization]
  // CHECK-FIXES: Agg d{1, 2};
}

void aggregate_default_member() {
  AggDefault ad(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use braced initialization
  // CHECK-FIXES: AggDefault ad{1, 2};
}

void nested_aggregate_braced_inner() {
  Nested n(Agg{1, 2}, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Nested n{Agg{1, 2}, 3};
}

// The inner parenthesized aggregate is a temporary expression, not a
// declaration, so only the outer variable declaration is rewritten.
void nested_aggregate_paren_inner() {
  Nested n(Agg(1, 2), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Nested n{Agg(1, 2), 3};
}

void aggregate_multi_decl() {
  Agg a(1, 2), b(3, 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use braced initialization
  // CHECK-FIXES: Agg a{1, 2}, b{3, 4};
}

void designated_as_arg() {
  Takes t({.a = 1, .b = 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: Takes t{{[{][{]}}.a = 1, .b = 2{{[}][}]}};
}

struct L1 {
  int a, b;
};

struct L2 {
  L1 x;
  int y;
};

// Only the outer declaration is rewritten; the inner paren aggregate is a
// temporary and is left untouched.
void nested_agg_two_levels() {
  L2 v(L1(1, 2), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-FIXES: L2 v{L1(1, 2), 3};
}

void already_braced() {
  Agg d{1, 2};
}

void copy_init() {
  Agg d = {1, 2};
}

void designated_already_braced() {
  Agg d{.a = 1, .b = 2};
}

void designated_copy_init() {
  Agg d = {.a = 1, .b = 2};
}

// Narrowing conversions in aggregate paren init: fix-it attached to the note.
struct AggFloat {
  int x;
  int y;
};

void narrowing_aggregate() {
  AggFloat af(1, 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:18: note: narrowing conversion from 'double' to 'int'
  // CHECK-FIXES: AggFloat af{1, 3.14};
}

void narrowing_aggregate_multiple() {
  AggFloat af(2.5, 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:15: note: narrowing conversion from 'double' to 'int'
  // CHECK-MESSAGES: :[[@LINE-3]]:20: note: narrowing conversion from 'double' to 'int'
  // CHECK-FIXES: AggFloat af{2.5, 3.14};
}

void no_narrowing_aggregate() {
  Agg a(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: Agg a{1, 2};
}

void array_paren_init() {
  int a[3](1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int a[3]{1, 2, 3};
}

void array_paren_init_unsized() {
  int a[](1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int a[]{1, 2, 3};
}

void range_for_with_init() {
  int arr[] = {1, 2, 3};
  for (Simple s(1); auto x : arr) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
    // CHECK-FIXES: for (Simple s{1}; auto x : arr) {
  }
}

// Lambda init-captures are not variable declarations and are left untouched.
void lambda_capture_init() {
  auto f = [s = Simple(1)](){};
}

void lambda_capture_aggregate() {
  auto f = [a = Agg(1, 2)](){};
}

void ctad_no_warn() {
  std::vector v(5, 1);
}
