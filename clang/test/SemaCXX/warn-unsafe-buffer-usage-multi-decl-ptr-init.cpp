// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s

void lhs_span_multi_assign() {
  int *a = new int[2];
  int *b = a;
  int *c = b;
  int *d = c;  // expected-warning{{'d' is an unsafe pointer used for buffer access}} expected-note{{change type of 'd' to 'std::span' to preserve bounds information, and change 'c', 'b', and 'a' to 'std::span' to propagate bounds information between them}}
  int tmp = d[2];  // expected-note{{used in buffer access here}}
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
  int *p = q;  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' to 'std::span' to propagate bounds information between them}}
  p[5] = 10;  // expected-note{{used in buffer access here}}
}

// FIXME: Suggest fixits for p, q, and r since span a valid fixit for r.
void rhs_span3() {
  int *q = new int[6];
  int *p = q;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p[5] = 10;  // expected-note{{used in buffer access here}}
  int *r = q;
}

void test_grouping() {
  int *z = new int[8];
  int tmp;
  int *y = new int[10];  // expected-warning{{'y' is an unsafe pointer used for buffer access}} expected-note{{change type of 'y' to 'std::span' to preserve bounds information}}
  tmp = y[5]; // expected-note{{used in buffer access here}}

  int *x = new int[10];
  x = y;

  int *w = z;
}

void test_crash() {
  int *r = new int[8];
  int *q = r;
  int *p;  // expected-warning{{'p' is an unsafe pointer used for buffer access}} expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to 'std::span' to propagate bounds information between them}}
  p = q;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
}
