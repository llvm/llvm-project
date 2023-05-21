// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void basic_dereference() {
  int tmp;
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  tmp = p[5];
  int val = *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:14}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"[0]"
}

int return_method() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  int tmp = p[5];
  return *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:11}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"[0]"
}

void foo(int v) {
}

void method_invocation() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];

  foo(*p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:9-[[@LINE-2]]:9}:"[0]"
}

void binary_operation() {
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];

  int k = *p + 20;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:12}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"[0]"

}

