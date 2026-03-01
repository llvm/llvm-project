// RUN: %check_clang_tidy -std=c++20-or-later %s misc-use-braced-initialization %t

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

void nested_aggregate_paren_inner() {
  Nested n(Agg(1, 2), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use braced initialization
  // CHECK-FIXES: Nested n{Agg{1, 2}, 3};
}

void aggregate_multi_decl() {
  Agg a(1, 2), b(3, 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use braced initialization
  // CHECK-FIXES: Agg a{1, 2}, b{3, 4};
}

void aggregate_temporary() {
  Agg(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use braced initialization
  // CHECK-FIXES: Agg{1, 2};
}

void aggregate_temporary_cast_to_void() {
  (void)Agg(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)Agg{1, 2};
}

void aggregate_auto() {
  auto d = Agg(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: auto d = Agg{1, 2};
}

Agg return_aggregate() {
  return Agg(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: return Agg{1, 2};
}

void func_arg(Agg);
void aggregate_as_argument() {
  func_arg(Agg(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: func_arg(Agg{1, 2});
}

void designated_as_arg() {
  Takes t({.a = 1, .b = 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: Takes t{{[{][{]}}.a = 1, .b = 2{{[}][}]}};
}

void aggregate_new() {
  Agg *p = new Agg(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: Agg *p = new Agg{1, 2};
  (void)p;
}

void lambda_capture_init() {
  auto f = [s = Simple(1)](){};
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization
  // CHECK-FIXES: auto f = [s = Simple{1}](){};
}

void ternary_arg(bool c) {
  Simple s(c ? 1 : 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple s{c ? 1 : 2};
}

struct L1 {
  int a, b;
};

struct L2 {
  L1 x;
  int y;
};

struct L3 {
  L2 m;
  int z;
};

struct L4 {
  L3 n;
  int w;
};

void nested_agg_two_levels() {
  L2 v(L1(1, 2), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-FIXES: L2 v{L1{1, 2}, 3};
}

void nested_agg_three_levels() {
  L3 v(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void nested_agg_four_levels() {
  L4 v(L3(L2(L1(1, 2), 3), 4), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-4]]:14: warning: use braced initialization
  // CHECK-FIXES: L4 v{L3{L2{L1{1, 2}, 3}, 4}, 5};
}

void nested_agg_temporary() {
  (void)L3(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:15: warning: use braced initialization
  // CHECK-FIXES: (void)L3{L2{L1{1, 2}, 3}, 4};
}

void nested_agg_new() {
  L3 *p = new L3(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: use braced initialization
  // CHECK-FIXES: L3 *p = new L3{L2{L1{1, 2}, 3}, 4};
  (void)p;
}

// Mixed: some levels already braced, only paren levels get fixed.
void nested_agg_mixed() {
  L3 v(L2{L1(1, 2), 3}, 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void nested_agg_mixed_inner_braced() {
  L3 v(L2(L1{1, 2}, 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void already_braced() {
  Agg d{1, 2};
}

void already_braced_temporary() {
  Agg{1, 2};
}

void new_already_braced() {
  Agg *p = new Agg{1, 2};
  (void)p;
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
