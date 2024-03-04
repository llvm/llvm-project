// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void unsafe_array_func_ptr_call(void (*fn_ptr)(int *param)) {
  int p[32];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::array<int, 32> p"

  int idx;
  p[idx] = 10;
  fn_ptr(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:11}:".data()"
}

void unsafe_ptr_func_ptr_call(void (*fn_ptr)(int *param)) {
  int *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "

  p[5] = 10;
  fn_ptr(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:11}:".data()"
}

void addr_of_unsafe_ptr_func_ptr_call(void (*fn_ptr)(int *param)) {
  int *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "

  p[5] = 10;
  fn_ptr(&p[0]);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:15}:"p.data()"
}

void addr_of_unsafe_ptr_w_offset_func_ptr_call(void (*fn_ptr)(int *param)) {
  int *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "

  p[5] = 10;
  fn_ptr(&p[3]);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:15}:"&p.data()[3]"
}

void preincrement_unsafe_ptr_func_ptr_call(void (*fn_ptr)(int *param)) {
  int *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "

  p[5] = 10;
  fn_ptr(++p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:13}:"(p = p.subspan(1)).data()"
}
