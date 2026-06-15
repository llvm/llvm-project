// RUN: %clang_cc1 -emit-llvm %s -std=c++2a -triple x86_64-unknown-linux-gnu -fexperimental-new-constant-interpreter -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -std=c++2a -triple x86_64-unknown-linux-gnu                                         -o - | FileCheck %s


struct Agg {
  int a;
  long b;
};
consteval Agg is_const(...) {
  return {5, 19 * __builtin_is_constant_evaluated()};
}
// CHECK-LABEL: @_Z13test_is_constv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[B:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_AGG:%.*]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    store i32 5, ptr [[TMP0]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT:    store i64 19, ptr [[TMP1]], align 8
// CHECK-NEXT:    store i64 19, ptr [[B]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[B]], align 8
// CHECK-NEXT:    ret i64 [[TMP2]]
long test_is_const() {
  long b = is_const().b;
  return b;
}
