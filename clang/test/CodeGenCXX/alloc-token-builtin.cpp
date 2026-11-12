// To test IR generation of the builtin without evaluating the LLVM intrinsic,
// we set the mode to a stateful mode, which prohibits constant evaluation.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Werror -std=c++20 -emit-llvm -falloc-token-mode=random -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-CODEGEN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Werror -std=c++20 -emit-llvm -falloc-token-max=2 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LOWER

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
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_INT:[0-9]+]])
// CHECK-LOWER: ret i64 0
unsigned long test_builtin_int() {
  return __builtin_infer_alloc_token(sizeof(1));
}

// CHECK-LABEL: @_Z16test_builtin_ptrv(
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_PTR:[0-9]+]])
// CHECK-LOWER: ret i64 1
unsigned long test_builtin_ptr() {
  return __builtin_infer_alloc_token(sizeof(int *));
}

// CHECK-LABEL: @_Z25test_builtin_struct_noptrv(
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_NOPTR:[0-9]+]])
// CHECK-LOWER: ret i64 0
unsigned long test_builtin_struct_noptr() {
  return __builtin_infer_alloc_token(sizeof(NoPtr));
}

// CHECK-LABEL: @_Z25test_builtin_struct_w_ptrv(
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_WITHPTR:[0-9]+]])
// CHECK-LOWER: ret i64 1
unsigned long test_builtin_struct_w_ptr() {
  return __builtin_infer_alloc_token(sizeof(WithPtr), 123);
}

// CHECK-LABEL: @_Z24test_builtin_unevaluatedv(
// CHECK-NOT: call{{.*}}unevaluated_fn
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_INT:[0-9]+]])
// CHECK-LOWER: ret i64 0
unsigned long test_builtin_unevaluated() {
	return __builtin_infer_alloc_token(sizeof(int) * unevaluated_fn());
}

// CHECK-LABEL: @_Z36test_builtin_unsequenced_unevaluatedi(
// CHECK:     add nsw
// CHECK-NOT: add nsw
// CHECK-CODEGEN: %[[REG:[0-9]+]] = call i64 @llvm.alloc.token.id.i64(metadata ![[META_UNKNOWN:[0-9]+]])
// CHECK-CODEGEN: call{{.*}}@my_malloc({{.*}}, i64 noundef %[[REG]])
// CHECK-LOWER: call{{.*}}@my_malloc({{.*}}, i64 noundef 0)
void test_builtin_unsequenced_unevaluated(int x) {
  my_malloc(++x, __builtin_infer_alloc_token(++x));
}

// CHECK-LABEL: @_Z20test_builtin_unknownv(
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_UNKNOWN:[0-9]+]])
// CHECK-LOWER: ret i64 0
unsigned long test_builtin_unknown() {
  return __builtin_infer_alloc_token(4096);
}

// Test template instantiation.
template <typename T>
constexpr unsigned long get_token() {
  return __builtin_infer_alloc_token(sizeof(T));
}

// CHECK-LABEL: @_Z13get_token_intv()
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_INT]])
// CHECK-LOWER: ret i64 0
unsigned long get_token_int() {
  return get_token<int>();
}

// CHECK-LABEL: @_Z13get_token_ptrv()
// CHECK-CODEGEN: call i64 @llvm.alloc.token.id.i64(metadata ![[META_PTR]])
// CHECK-LOWER: ret i64 1
unsigned long get_token_ptr() {
  return get_token<int *>();
}

// CHECK-CODEGEN: ![[META_INT]] = !{!"int", i1 false}
// CHECK-CODEGEN: ![[META_PTR]] = !{!"int *", i1 true}
// CHECK-CODEGEN: ![[META_NOPTR]] = !{!"NoPtr", i1 false}
// CHECK-CODEGEN: ![[META_WITHPTR]] = !{!"WithPtr", i1 true}
// CHECK-CODEGEN: ![[META_UNKNOWN]] = !{}
