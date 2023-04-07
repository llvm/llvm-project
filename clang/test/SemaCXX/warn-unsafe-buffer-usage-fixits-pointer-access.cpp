// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// XFAIL: *

void foo(int* v) {
}

void m1(int* v1, int* v2, int) {

}

void condition_check_nullptr() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  if(p != nullptr) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
}

void condition_check() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  auto q = new int[10];

  int tmp = p[5];
  if(p == q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(q != p){}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:12}:".data()"

  if(p < q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p <= q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p > q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p >= q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if( p == q && p != nullptr) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:".data()"
}

void condition_check_two_usafe_buffers() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  auto q = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  tmp = q[5];

  if(p == q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:".data()"
}

void unsafe_method_invocation_single_param() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  foo(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"

}

void safe_method_invocation_single_param() {
  auto p = new int[10];
  foo(p);
}

void unsafe_method_invocation_double_param() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  m1(p, p, 10);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:".data()"

  auto q = new int[10];

  m1(p, q, 4);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  m1(q, p, 9);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:10}:".data()"

  m1(q, q, 8);
}

