// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - -x c %s | FileCheck %s --check-prefix=CHECK-GNU-C
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-GNU-CXX

// Empty structs are ignored for PCS purposes on Darwin and in C mode elsewhere.
// In C++ mode on ELF they consume a register slot though. Functions are
// slightly bigger than minimal to make confirmation against actual GCC
// behaviour easier.

#if __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

struct Empty {};

// CHECK: define i32 @empty_arg(i32 %a)
// CHECK-GNU-C: define i32 @empty_arg(i32 %a)
// CHECK-GNU-CXX: define i32 @empty_arg(i8 %e.coerce, i32 %a)
EXTERNC int empty_arg(struct Empty e, int a) {
  return a;
}

// CHECK: define void @empty_ret()
// CHECK-GNU-C: define void @empty_ret()
// CHECK-GNU-CXX: define void @empty_ret()
EXTERNC struct Empty empty_ret() {
  struct Empty e;
  return e;
}

// However, what counts as "empty" is a baroque mess. This is super-empty, it's
// ignored even in C++ mode. It also has sizeof == 0, violating C++, but that's
// legacy for you:

struct SuperEmpty {
  int arr[0];
};

// CHECK: define i32 @super_empty_arg(i32 %a)
// CHECK-GNU-C: define i32 @super_empty_arg(i32 %a)
// CHECK-GNU-CXX: define i32 @super_empty_arg(i32 %a)
EXTERNC int super_empty_arg(struct SuperEmpty e, int a) {
  return a;
}

// This is not empty. It has 0 size but consumes a register slot for GCC.

struct SortOfEmpty {
  struct SuperEmpty e;
};

// CHECK: define i32 @sort_of_empty_arg(i32 %a)
// CHECK-GNU-C: define i32 @sort_of_empty_arg(i32 %a)
// CHECK-GNU-CXX: define i32 @sort_of_empty_arg(i8 %e.coerce, i32 %a)
EXTERNC int sort_of_empty_arg(struct Empty e, int a) {
  return a;
}

// CHECK: define void @sort_of_empty_ret()
// CHECK-GNU-C: define void @sort_of_empty_ret()
// CHECK-GNU-CXX: define void @sort_of_empty_ret()
EXTERNC struct SortOfEmpty sort_of_empty_ret() {
  struct SortOfEmpty e;
  return e;
}
