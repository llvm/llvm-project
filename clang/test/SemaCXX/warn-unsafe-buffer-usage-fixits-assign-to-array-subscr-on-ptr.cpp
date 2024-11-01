// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// TODO cases where we don't want fixits

// The Fix-It for unsafe operation is trivially empty.
// In order to test that our machinery recognizes that we can test if the variable declaration gets a Fix-It.
// If the operation wasn't handled propertly the declaration won't get Fix-It.
// By testing presence of the declaration Fix-It we indirectly test presence of the trivial Fix-It for its operations.
void test() {
  int *p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  p[5] = 1;
  // CHECK-NOT: fix-it:
}
