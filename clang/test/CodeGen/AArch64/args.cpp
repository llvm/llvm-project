// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - -x c %s | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CXX

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

// DARWIN: define{{.*}} i32 @empty_arg(i32 noundef %a)
// C: define{{.*}} i32 @empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @empty_arg(i8 %e.coerce, i32 noundef %a)
EXTERNC int empty_arg(struct Empty e, int a) {
  return a;
}

// DARWIN: define{{.*}} void @empty_ret()
// C: define{{.*}} void @empty_ret()
// CXX: define{{.*}} void @empty_ret()
EXTERNC struct Empty empty_ret(void) {
  struct Empty e;
  return e;
}

// However, what counts as "empty" is a baroque mess. This is super-empty, it's
// ignored even in C++ mode. It also has sizeof == 0, violating C++, but that's
// legacy for you:

struct SuperEmpty {
  int arr[0];
};

// DARWIN: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
// C: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
EXTERNC int super_empty_arg(struct SuperEmpty e, int a) {
  return a;
}

// This is also not empty, and non-standard. We previously considered it to
// consume a register slot, but GCC does not, so we match that.

struct SortOfEmpty {
  struct SuperEmpty e;
};

// DARWIN: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
// C: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
EXTERNC int sort_of_empty_arg(struct SortOfEmpty e, int a) {
  return a;
}

// DARWIN: define{{.*}} void @sort_of_empty_ret()
// C: define{{.*}} void @sort_of_empty_ret()
// CXX: define{{.*}} void @sort_of_empty_ret()
EXTERNC struct SortOfEmpty sort_of_empty_ret(void) {
  struct SortOfEmpty e;
  return e;
}

#include <stdarg.h>

// va_arg matches the above rules, consuming an incoming argument in cases
// where one would be passed, and not doing so when the argument should be
// ignored.

EXTERNC struct Empty empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @empty_arg_variadic(
// DARWIN-NOT: {{ getelementptr }}
// C-NOT: {{ getelementptr }}
// CXX: %new_reg_offs = add i32 %gr_offs, 8
// CXX: %new_stack = getelementptr inbounds i8, ptr %stack, i64 8
  va_list vl;
  va_start(vl, a);
  struct Empty b = va_arg(vl, struct Empty);
  va_end(vl);
  return b;
}

EXTERNC struct SuperEmpty super_empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @super_empty_arg_variadic(
// DARWIN-NOT: {{ getelementptr }}
// C-NOT: {{ getelementptr }}
// CXX-NOT: {{ getelementptr }}
  va_list vl;
  va_start(vl, a);
  struct SuperEmpty b = va_arg(vl, struct SuperEmpty);
  va_end(vl);
  return b;
}

EXTERNC struct SortOfEmpty sort_of_empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @sort_of_empty_arg_variadic(
// DARWIN: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 0
// C-NOT: {{ getelementptr }}
// CXX-NOT: {{ getelementptr }}
  va_list vl;
  va_start(vl, a);
  struct SortOfEmpty b = va_arg(vl, struct SortOfEmpty);
  va_end(vl);
  return b;
}

