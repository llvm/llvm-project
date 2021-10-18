// RUN: %clang -cc1 -x c++ -triple x86_64-unknown-unknown -O0 -emit-llvm -enable-noundef-analysis -o - %s | FileCheck %s

union u1 {
  int val;
};

// CHECK: @indirect_callee_int_ptr = [[GLOBAL:(dso_local )?global]] i32 (i32)*
int (*indirect_callee_int_ptr)(int);
// CHECK: @indirect_callee_union_ptr = [[GLOBAL]] i32 (i32)*
union u1 (*indirect_callee_union_ptr)(union u1);

// CHECK: [[DEFINE:define( dso_local)?]] noundef i32 @{{.*}}indirect_callee_int{{.*}}(i32 noundef %
int indirect_callee_int(int a) { return a; }
// CHECK: [[DEFINE]] i32 @{{.*}}indirect_callee_union{{.*}}(i32 %
union u1 indirect_callee_union(union u1 a) {
  return a;
}

int main() {
  // CHECK: call noundef i32 @{{.*}}indirect_callee_int{{.*}}(i32 noundef 0)
  indirect_callee_int(0);
  // CHECK: call i32 @{{.*}}indirect_callee_union{{.*}}(i32 %
  indirect_callee_union((union u1){0});

  indirect_callee_int_ptr = indirect_callee_int;
  indirect_callee_union_ptr = indirect_callee_union;

  // CHECK: call noundef i32 %{{.*}}(i32 noundef 0)
  indirect_callee_int_ptr(0);
  // CHECK: call i32 %{{.*}}(i32 %
  indirect_callee_union_ptr((union u1){});

  return 0;
}
