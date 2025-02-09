// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void safe_array_initing_safe_ptr(unsigned idx) {
  int buffer[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int* ptr = buffer;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
}

void safe_array_initing_unsafe_ptr(unsigned idx) {
  int buffer[123321123];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int* ptr = buffer;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}123321123
  ptr[idx + 1] = 0;
}
