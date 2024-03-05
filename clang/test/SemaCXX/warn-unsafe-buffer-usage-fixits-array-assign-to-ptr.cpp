// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void safe_array_assigned_to_safe_ptr(unsigned idx) {
  int buffer[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int* ptr;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  ptr = buffer;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
}

void safe_array_assigned_to_unsafe_ptr(unsigned idx) {
  int buffer[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int* ptr;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  ptr = buffer;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  ptr[idx] = 0;
}

void unsafe_array_assigned_to_safe_ptr(unsigned idx) {
  int buffer[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 10> buffer"
  int* ptr;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  ptr = buffer;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:15-[[@LINE-1]]:15}:".data()"
  buffer[idx] = 0;
}

// FIXME: Implement fixit/s for this case.
// See comment in CArrayToPtrAssignmentGadget::getFixits to learn why this hasn't been implemented.
void unsafe_array_assigned_to_unsafe_ptr(unsigned idx) {
  int buffer[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}
  int* ptr;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}
  ptr = buffer;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}
  buffer[idx] = 0;
  ptr[idx] = 0;
}
