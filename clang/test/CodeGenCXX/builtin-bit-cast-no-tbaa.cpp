// RUN: %clang_cc1 -O3 -std=c++2a -S -emit-llvm -o - -disable-llvm-passes -triple x86_64-apple-macos10.14 %s | FileCheck %s

void test_scalar() {
  // CHECK-LABEL: define{{.*}} void @_Z11test_scalarv
  __builtin_bit_cast(float, 42);

  // CHECK: load float, float* {{.*}}, align 4, !tbaa ![[MAY_ALIAS_TBAA:.*]]
}

void test_scalar2() {
  // CHECK-LABEL: define{{.*}} void @_Z12test_scalar2v
  struct S {int m;};
  __builtin_bit_cast(int, S{42});

  // CHECK: load i32, i32* {{.*}}, align 4, !tbaa ![[MAY_ALIAS_TBAA]]
}

int test_same_type(int &r) {
  // CHECK: load i32, i32* {{.*}}, align 4, !tbaa ![[MAY_ALIAS_TBAA]]
  return __builtin_bit_cast(int, r);
}

// CHECK: ![[CHAR_TBAA:.*]] = !{!"omnipotent char", {{.*}}, i64 0}
// CHECK: ![[MAY_ALIAS_TBAA]] = !{![[CHAR_TBAA]], ![[CHAR_TBAA]], i64 0}
