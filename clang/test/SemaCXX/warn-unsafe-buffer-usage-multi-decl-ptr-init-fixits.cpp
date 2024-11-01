// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void lhs_span_multi_assign() {
  int *a = new int[2];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> a"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 2}"
  int *b = a;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> b"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  int *c = b;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> c"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  int *d = c;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> d"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  int tmp = d[2];  // expected-note{{used in buffer access here}}
}

void rhs_span1() {
  int *q = new int[12];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 12}"
  int *p = q;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  p[5] = 10;
  int *r = q;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  r[10] = 5;  // expected-note{{used in buffer access here}}
}

void rhs_span2() {
  int *q = new int[6];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 6}"
  int *p = q;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  p[5] = 10;  // expected-note{{used in buffer access here}}
}

void rhs_span3() {
  int *q = new int[6];
  int *p = q;  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p[5] = 10;  // expected-note{{used in buffer access here}}
  int *r = q;
}

void test_grouping() {
  int *z = new int[8];
  int tmp;
  int *y = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> y"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  tmp = y[5];

  int *x = new int[10];
  x = y;

  int *w = z;
}

void test_crash() {
  int *r = new int[8];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 8}"
  int *q = r;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  int *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:9}:"std::span<int> p"
  p = q;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
}
