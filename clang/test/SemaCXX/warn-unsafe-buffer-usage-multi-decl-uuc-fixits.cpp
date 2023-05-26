// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void bar(int * param) {}

void foo1a() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = r;
  int tmp = p[9];
  int *q;
  q = r;
}

void uuc_if_body() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  if (true)
    p = r;
  p[5] = 4;
}

void uuc_if_body1(bool flag) {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  if (flag) {
    p = r;
  }
  p[5] = 4;
}

void uuc_if_cond_no_unsafe_op() {
  int *r = new int[7];
  int *p = new int[4];
  if ((p = r)) {
    int x = 0;
  }
}

void uuc_if_cond_unsafe_op() {
  int *r = new int[7];
  int *p = new int[4];  //expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if ((p = r)) {
    p[3] = 2;  // expected-note{{used in buffer access here}}
  }
}

void uuc_if_cond_unsafe_op1() {
  int *r = new int[7];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *p = new int[4];
  if ((p = r)) {
    r[3] = 2;  // expected-note{{used in buffer access here}}
  }
}

void uuc_if_cond_unsafe_op2() {
  int *r = new int[7];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if ((p = r)) {
    r[3] = 2;  // expected-note{{used in buffer access here}}
  }
  p[4] = 6;  // expected-note{{used in buffer access here}}
}

void uuc_call1() {
  int *w = new int[4];  // expected-warning{{'w' is an unsafe pointer used for buffer access}}
  int *y = new int[4];
  bar(w = y);
  w[5] = 0;  // expected-note{{used in buffer access here}}
}
