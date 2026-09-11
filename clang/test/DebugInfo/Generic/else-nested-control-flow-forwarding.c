// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

// Nested control flow at the end of an else body can leave behind an empty
// continuation block that only forwards to the outer if continuation. Avoid
// emitting that extra block.

// CHECK-LABEL: define {{.*}} @else_if_chain(
// CHECK: if.end:
// CHECK-NOT: if.end{{[0-9]+}}:
// CHECK: ret

int else_if_chain(int i, int *arr) {
  int x = 0;
  if (i == 1) {
    x = i + 1;
  } else if (i == 2) {
    arr[i] = 42;
  }
  return x;
}

// CHECK-LABEL: define {{.*}} @nested_if_in_else(
// CHECK: if.end:
// CHECK-NOT: if.end{{[0-9]+}}:
// CHECK: ret

int nested_if_in_else(int i, int *arr) {
  int x = 0;
  if (i == 1) {
    x = i + 1;
  } else {
    if (i == 2) {
      arr[i] = 42;
    }
  }
  return x;
}

// CHECK-LABEL: define {{.*}} @else_if_chain_with_final_else(
// CHECK: if.end:
// CHECK-NOT: if.end{{[0-9]+}}:
// CHECK: ret

int else_if_chain_with_final_else(int i) {
  int x = 0;
  if (i == 1) {
    x = 1;
  } else if (i == 2) {
    x = 2;
  } else if (i == 3) {
    x = 3;
  } else {
    x = 4;
  }
  return x;
}
