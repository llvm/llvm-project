// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void bar(int * param) {}

void foo1a() {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = r;
  int tmp = p[9];
  int *q;
  q = r;
}

void uuc_if_body() {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  if (true)
    p = r;
  p[5] = 4;
}

void uuc_if_body1(bool flag) {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  if (flag) {
    p = r;
  }
  p[5] = 4;
}

void uuc_if_body2_ptr_init(bool flag) {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  if (flag) {
  } else {
    int* p = r;
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:13}:"std::span<int> p"
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"{"
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:15}:", <# placeholder #>}"
    p[5] = 4;
  }
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
  int *p = new int[4];
  if ((p = r)) {
    p[3] = 2;
  }
}

void uuc_if_cond_unsafe_op1() {
  int *r = new int[7];
  int *p = new int[4];
  if ((p = r)) {
    r[3] = 2;
  }
}

void uuc_if_cond_unsafe_op2() {
  int *r = new int[7];
  int *p = new int[4];
  if ((p = r)) {
    r[3] = 2;
  }
  p[4] = 6;
}

void uuc_call1() {
  int *w = new int[4];
  int *y = new int[4];
  bar(w = y);
  w[5] = 0;
}
