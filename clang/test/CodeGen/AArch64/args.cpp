// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - -x c %s | FileCheck %s --check-prefixes=CHECK,C,AAPCS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CXX,AAPCS

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

// Base case, nothing interesting.
struct S {
  long x, y;
};

// CHECK-LABEL: @g_S(
// CHECK: call void @f_S(i64 noundef 1, [2 x i64] {{.*}})
// CHECK: call void @fm_S(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, [2 x i64] {{.*}})
EXTERNC void f_S(long, struct S);
EXTERNC void fm_S(long, long, long, long, long, struct S);
EXTERNC void g_S() {
  struct S s = {6, 7};
  f_S(1, s);
  fm_S(1, 2, 3, 4, 5, s);
}

// Aligned struct passed according to its natural alignment.
struct __attribute__((aligned(16))) S16 {
  long x, y;
};

// CHECK-LABEL: @g_S16(
// DARWIN: call void @f_S16(i64 noundef 1, i128 {{.*}})
// AAPCS: call void @f_S16(i64 noundef 1, [2 x i64] {{.*}})
// DARWIN: call void @fm_S16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 {{.*}})
// AAPCS: call void @fm_S16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, [2 x i64] {{.*}})
EXTERNC void f_S16(long, struct S16);
EXTERNC void fm_S16(long, long, long, long, long, struct S16);
EXTERNC void g_S16() {
  struct S16 s = {6, 7};
  f_S16(1, s);
  fm_S16(1, 2, 3, 4, 5, s);
}

// Aligned struct with increased natural alignment through an aligned field.
struct SF16 {
  __attribute__((aligned(16))) long x;
  long y;
};

// CHECK-LABEL: @g_SF16(
// DARWIN: call void @f_SF16(i64 noundef 1, i128 {{.*}})
// AAPCS: call void @f_SF16(i64 noundef 1, i128 {{.*}})
// DARWIN: call void @fm_SF16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 {{.*}})
// AAPCS: call void @fm_SF16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 {{.*}})
EXTERNC void f_SF16(long, struct SF16);
EXTERNC void fm_SF16(long, long, long, long, long, struct SF16);
EXTERNC void g_SF16() {
  struct SF16 s = {6, 7};
  f_SF16(1, s);
  fm_SF16(1, 2, 3, 4, 5, s);
}

#ifdef __cplusplus
// Aligned struct with increased natural alignment through an aligned base class.
struct SB16 : S16 {};

// DARWIN-LABEL: @g_SB16(
// CXX-LABEL: @g_SB16(
// DARWIN: call void @f_SB16(i64 noundef 1, i128 {{.*}})
// CXX: call void @f_SB16(i64 noundef 1, i128 {{.*}})
// DARWIN: call void @fm_SB16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 {{.*}})
// CXX: call void @fm_SB16(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 {{.*}})
EXTERNC void f_SB16(long, struct SB16);
EXTERNC void fm_SB16(long, long, long, long, long, struct SB16);
EXTERNC void g_SB16() {
  struct SB16 s = {6, 7};
  f_SB16(1, s);
  fm_SB16(1, 2, 3, 4, 5, s);
}
#endif

// Packed structure.
struct  __attribute__((packed)) SP {
  int x;
  long y;
};

// CHECK-LABEL: @g_SP(
// CHECK: call void @f_SP(i32 noundef 1, [2 x i64] {{.*}})
// CHECK: call void @fm_SP(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i64] {{.*}})
EXTERNC void f_SP(int, struct SP);
EXTERNC void fm_SP(int, int, int, int, int, struct SP);
EXTERNC void g_SP() {
  struct SP s = {6, 7};
  f_SP(1, s);
  fm_SP(1, 2, 3, 4, 5, s);
}

// Packed structure, overaligned, same as above.
struct  __attribute__((packed, aligned(16))) SP16 {
  int x;
  long y;
};

// CHECK-LABEL: @g_SP16(
// DARWIN: call void @f_SP16(i32 noundef 1, i128 {{.*}})
// AAPCS: call void @f_SP16(i32 noundef 1, [2 x i64] {{.*}})
// DARWIN: call void @fm_SP16(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i128 {{.*}})
// AAPCS: call void @fm_SP16(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i64] {{.*}})
EXTERNC void f_SP16(int, struct SP16);
EXTERNC void fm_SP16(int, int, int, int, int, struct SP16);
EXTERNC void g_SP16() {
  struct SP16 s = {6, 7};
  f_SP16(1, s);
  fm_SP16(1, 2, 3, 4, 5, s);
}
