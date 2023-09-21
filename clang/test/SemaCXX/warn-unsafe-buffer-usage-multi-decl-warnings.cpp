// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s
namespace std {
  class type_info { };
}

void local_assign_both_span() {
  int tmp;
  int* p = new int[10]; // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' to 'std::span' to propagate bounds information between them}}
  tmp = p[4];  // expected-note{{used in buffer access here}}

  int* q = new int[10];  // expected-warning{{'q' is an unsafe pointer used for buffer access}} expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p' to 'std::span' to propagate bounds information between them}}
  tmp = q[4];  // expected-note{{used in buffer access here}}

  q = p;
}

void local_assign_rhs_span() {
  int tmp;
  int* p = new int[10];
  int* q = new int[10];  // expected-warning{{'q' is an unsafe pointer used for buffer access}}
  tmp = q[4];  // expected-note{{used in buffer access here}}
  p = q;  // FIXME: we do not fix `p = q` here as the `.data()` fix-it is not generally correct
}

void local_assign_no_span() {
  int tmp;
  int* p = new int[10];
  int* q = new int[10];
  p = q;
}

void local_assign_lhs_span() {
  int tmp;
  int* p = new int[10];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' to 'std::span' to propagate bounds information between them}}
  tmp = p[4];  // expected-note{{used in buffer access here}}
  int* q = new int[10];

  p = q;
}

void lhs_span_multi_assign() {
  int *a = new int[2];
  int *b = a;
  int *c = b;
  int *d = c;  // expected-warning{{'d' is an unsafe pointer used for buffer access}} expected-note{{change type of 'd' to 'std::span' to preserve bounds information, and change 'c', 'b', and 'a' to 'std::span' to propagate bounds information between them}}
  int tmp = d[2];  // expected-note{{used in buffer access here}}
}

void rhs_span() {
  int *x = new int[3];
  int *y;  // expected-warning{{'y' is an unsafe pointer used for buffer access}}
  y[5] = 10;  // expected-note{{used in buffer access here}}

  x = y; // FIXME: we do not fix `x = y` here as the `.data()` fix-it is not generally correct
}

void rhs_span1() {
  int *q = new int[12];
  int *p = q;  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  p[5] = 10;  // expected-note{{used in buffer access here}}
  int *r = q;  // expected-warning{{'r' is an unsafe pointer used for buffer access}} expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p' and 'q' to 'std::span' to propagate bounds information between them}}
  r[10] = 5;  // expected-note{{used in buffer access here}}
}

void rhs_span2() {
  int *q = new int[6];
  int *p = q; // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p[5] = 10;  // expected-note{{used in buffer access here}}
  int *r = q; // FIXME: we do not fix `int *r = q` here as the `.data()` fix-it is not generally correct
}

void test_grouping() {
  int *z = new int[8];
  int tmp;
  int *y = new int[10];  // expected-warning{{'y' is an unsafe pointer used for buffer access}}
  tmp = y[5]; // expected-note{{used in buffer access here}}

  int *x = new int[10];
  x = y;      // FIXME: we do not fix `x = y` here as the `.data()` fix-it is not generally correct

  int *w = z;
}

void test_grouping1() {
  int tmp;
  int *y = new int[10];  // expected-warning{{'y' is an unsafe pointer used for buffer access}}
  tmp = y[5];  // expected-note{{used in buffer access here}}
  int *x = new int[10];
  x = y;       // FIXME: we do not fix `x = y` here as the `.data()` fix-it is not generally correct

  int *w = new int[10];  // expected-warning{{'w' is an unsafe pointer used for buffer access}}
  tmp = w[5];  // expected-note{{used in buffer access here}}
  int *z = new int[10];
  z = w;       // FIXME: we do not fix `z = w` here as the `.data()` fix-it is not generally correct
}

void foo1a() {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p = r;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
  int *q;
  q = r;  // FIXME: we do not fix `q = r` here as the `.data()` fix-it is not generally correct
}

void foo1b() {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' and 'q' to 'std::span' to propagate bounds information between them}}
  p = r;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
  int *q;  // expected-warning{{'q' is an unsafe pointer used for buffer access}} expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p' and 'r' to 'std::span' to propagate bounds information between them}}
  q = r;
  tmp = q[9];  // expected-note{{used in buffer access here}}
}

void foo1c() {
  int *r = new int[7];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *p = new int[4];
  p = r;   // FIXME: we do not fix `p = r` here as the `.data()` fix-it is not generally correct
  int tmp = r[9];  // expected-note{{used in buffer access here}}
  int *q;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}
  q = r;   // FIXME: we do not fix `q = r` here as the `.data()` fix-it is not generally correct
  tmp = q[9];  // expected-note{{used in buffer access here}}
}

void foo2a() {
  int *r = new int[7];
  int *p = new int[5];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  int *q = new int[4];
  p = q;
  int tmp = p[8];  // expected-note{{used in buffer access here}}
  q = r;
}

void foo2b() {
  int *r = new int[7];
  int *p = new int[5];
  int *q = new int[4];  // expected-warning{{'q' is an unsafe pointer used for buffer access}}
  p = q;           // FIXME: we do not fix `p = q` here as the `.data()` fix-it is not generally correct
  int tmp = q[8];  // expected-note{{used in buffer access here}}
  q = r;
}

void foo2c() {
  int *r = new int[7];
  int *p = new int[5];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  int *q = new int[4];  // expected-warning{{'q' is an unsafe pointer used for buffer access}} expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p' and 'r' to 'std::span' to propagate bounds information between them}}
  p = q;
  int tmp = p[8];  // expected-note{{used in buffer access here}}
  q = r;
  tmp = q[8];  // expected-note{{used in buffer access here}}
}

void foo3a() {
  int *r = new int[7];
  int *p = new int[5];  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  int *q = new int[4];
  q = p;           // FIXME: we do not fix `q = p` here as the `.data()` fix-it is not generally correct
  int tmp = p[8];  // expected-note{{used in buffer access here}}
  q = r;
}

void foo3b() {
  int *r = new int[7];
  int *p = new int[5];
  int *q = new int[4];  // expected-warning{{'q' is an unsafe pointer used for buffer access}} //expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'r' and 'p' to 'std::span' to propagate bounds information between them}}
  q = p;
  int tmp = q[8];  // expected-note{{used in buffer access here}}
  q = r;
}

void test_crash() {
  int *r = new int[8];
  int *q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  p = q;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
}

void foo_uuc() {
  int *ptr;
  int *local;  // expected-warning{{'local' is an unsafe pointer used for buffer access}}
  local = ptr;
  local++;  // expected-note{{used in pointer arithmetic here}}

  (local = ptr) += 5;  // expected-warning{{unsafe pointer arithmetic}}
}

void check_rhs_fix() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}  // expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'x' to 'std::span' to propagate bounds information between them}}
  int *x;
  r[7] = 9;  // expected-note{{used in buffer access here}}
  r = x;
}

void check_rhs_nofix() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  r = x;
  x++;  // expected-note{{used in pointer arithmetic here}}
}

void check_rhs_nofix_order() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  x++;  // expected-note{{used in pointer arithmetic here}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  r = x;
}

void check_rhs_nofix_order1() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  x++;  // expected-note{{used in pointer arithmetic here}}
  r = x;
}

void check_rhs_nofix_order2() {
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  x++;  // expected-note{{used in pointer arithmetic here}}
  r = x;
}

void check_rhs_nofix_order3() {
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r = x;
  r[7] = 9;  // expected-note{{used in buffer access here}}
  x++;  // expected-note{{used in pointer arithmetic here}}
}

void check_rhs_nofix_order4() {
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  r = x;
  x++;  // expected-note{{used in pointer arithmetic here}}
}

void no_unhandled_lhs() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}  // expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'x' to 'std::span' to propagate bounds information between them}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  int *x;
  r = x;
}

const std::type_info unhandled_lhs() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  int *x;
  r = x;
  return typeid(*r);
}

const std::type_info unhandled_rhs() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  r[7] = 9;  // expected-note{{used in buffer access here}}
  int *x;
  r = x;
  return typeid(*x);
}

void test_negative_index() {
  int *x = new int[4];  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p = &x[1];  // expected-note{{used in buffer access here}}
  p[-1] = 9;  // expected-note{{used in buffer access here}}
}

void test_unfixable() {
  int *r = new int[8];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *x;  // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  x[7] = 9;  // expected-note{{used in buffer access here}}
  r = x;
  r++;  // expected-note{{used in pointer arithmetic here}}
}

void test_cyclic_deps() {
  int *r = new int[10];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p' and 'q' to 'std::span' to propagate bounds information between them}}
  int *q;
  q = r;
  int *p;
  p = q;
  r[3] = 9; // expected-note{{used in buffer access here}}
  r = p;
}

void test_cyclic_deps_a() {
  int *r = new int[10];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *q;
  q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p = q;
  r[3] = 9; // expected-note{{used in buffer access here}}
  r = p;
  p++;  // expected-note{{used in pointer arithmetic here}}
}

void test_cyclic_deps1() {
  int *r = new int[10];
  int *q;
  q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  p = q;
  p[3] = 9; // expected-note{{used in buffer access here}}
  r = p;
}

void test_cyclic_deps2() {
  int *r = new int[10];
  int *q;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'r' and 'p' to 'std::span' to propagate bounds information between them}}
  q = r;
  int *p;
  p = q;
  q[3] = 9; // expected-note{{used in buffer access here}}
  r = p;
}

void test_cyclic_deps3() {
  int *r = new int[10];
  int *q;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'r' and 'p' to 'std::span' to propagate bounds information between them}}
  q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  p = q;
  q[3] = 9; // expected-note{{used in buffer access here}}
  p[4] = 7; // expected-note{{used in buffer access here}}
  r = p;
}

void test_cyclic_deps4() {
  int *r = new int[10];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p' and 'q' to 'std::span' to propagate bounds information between them}}
  int *q;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'r' and 'p' to 'std::span' to propagate bounds information between them}}
  q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}  expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' and 'q' to 'std::span' to propagate bounds information between them}}
  p = q;
  q[3] = 9; // expected-note{{used in buffer access here}}
  p[4] = 7; // expected-note{{used in buffer access here}}
  r[1] = 5; // expected-note{{used in buffer access here}}
  r = p;
}
