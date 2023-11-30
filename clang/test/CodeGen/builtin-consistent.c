// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm -disable-llvm-passes -o - %s -O1 | FileCheck %s 
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm -o - %s -O0 | FileCheck %s --check-prefix=CHECK_O0

void f(void);
void g(void);
void consistent_branch(int x) {
// CHECK-LABEL: define{{.*}} void @consistent_branch(
// CHECK-NOT: builtin_consistent
// CHECK: !consistent [[METADATA:.+]]
// CHECK_O0-NOT: builtin_consistent
// CHECK_O0-NOT: !consistent 
  if (__builtin_consistent(x > 0))
    f();
  
  if (x || __builtin_consistent(x != 0))
    g();
}

int consistent_switch(int x) {
// CHECK-LABEL: @consistent_switch(
// CHECK-NOT: builtin_consistent
// CHECK: !consistent [[METADATA:.+]]
// CHECK_O0-NOT: builtin_consistent
// CHECK_O0-NOT: !consistent 
  switch(__builtin_consistent(x)) {
  default:
    return x;
  case 0:
  case 1:
  case 2:
    return 1;
  case 3:
    return x-1;
  };
}
// CHECK: [[METADATA]] = !{i1 true}

