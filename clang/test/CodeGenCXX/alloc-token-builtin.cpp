// Test IR generation of the builtin without evaluating the LLVM intrinsic.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Werror -std=c++20 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-CODEGEN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Werror -std=c++20 -emit-llvm -falloc-token-max=2  %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LOWER

extern "C" void *my_malloc(unsigned long, unsigned long);

struct NoPtr {
  int x;
  long y;
};

struct WithPtr {
  int a;
  char *buf;
};

int unevaluated_fn();

// CHECK-LABEL: @_Z16test_builtin_intv(
unsigned long test_builtin_int() {
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_INT:[0-9]+]])
  // CHECK-LOWER: ret i64 0
  return __builtin_infer_alloc_token(sizeof(1));
}

// CHECK-LABEL: @_Z16test_builtin_ptrv(
unsigned long test_builtin_ptr() {
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_PTR:[0-9]+]])
  // CHECK-LOWER: ret i64 1
  return __builtin_infer_alloc_token(sizeof(int *));
}

// CHECK-LABEL: @_Z25test_builtin_struct_noptrv(
unsigned long test_builtin_struct_noptr() {
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_NOPTR:[0-9]+]])
  // CHECK-LOWER: ret i64 0
  return __builtin_infer_alloc_token(sizeof(NoPtr));
}

// CHECK-LABEL: @_Z25test_builtin_struct_w_ptrv(
unsigned long test_builtin_struct_w_ptr() {
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_WITHPTR:[0-9]+]])
  // CHECK-LOWER: ret i64 1
  return __builtin_infer_alloc_token(sizeof(WithPtr), 123);
}

// CHECK-LABEL: @_Z24test_builtin_unevaluatedv(
unsigned long test_builtin_unevaluated() {
  // CHECK-NOT: call{{.*}}unevaluated_fn
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_INT:[0-9]+]])
  // CHECK-LOWER: ret i64 0
  return __builtin_infer_alloc_token(sizeof(int) * unevaluated_fn());
}

// CHECK-LABEL: @_Z36test_builtin_unsequenced_unevaluatedi(
void test_builtin_unsequenced_unevaluated(int x) {
  // CHECK:     add nsw
  // CHECK-NOT: add nsw
  // CHECK-CODEGEN: %[[REG:[0-9]+]] = call i64 @llvm.alloc.token.id.i64(metadata ![[MD_UNKNOWN:[0-9]+]])
  // CHECK-CODEGEN: call{{.*}}@my_malloc({{.*}}, i64 noundef %[[REG]])
  // CHECK-LOWER: call{{.*}}@my_malloc({{.*}}, i64 noundef 0)
  my_malloc(++x, __builtin_infer_alloc_token(++x));
}

// CHECK-LABEL: @_Z20test_builtin_unknownv(
unsigned long test_builtin_unknown() {
  // CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[MD_UNKNOWN:[0-9]+]])
  // CHECK-LOWER: ret i64 0
  return __builtin_infer_alloc_token(4096);
}

// CHECK-CODEGEN: ![[MD_INT]] = !{!"int", i1 false}
// CHECK-CODEGEN: ![[MD_PTR]] = !{!"int *", i1 true}
// CHECK-CODEGEN: ![[MD_NOPTR]] = !{!"NoPtr", i1 false}
// CHECK-CODEGEN: ![[MD_WITHPTR]] = !{!"WithPtr", i1 true}
// CHECK-CODEGEN: ![[MD_UNKNOWN]] = !{}
