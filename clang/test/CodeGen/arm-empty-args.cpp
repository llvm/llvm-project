// RUN: %clang_cc1 -triple armv7a-linux-gnueabi -emit-llvm -o - -x c %s | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple armv7a-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CXX
// RUN: %clang_cc1 -triple armv7a-linux-gnueabi -emit-llvm -o - %s -fclang-abi-compat=19 | FileCheck %s --check-prefixes=CHECK,CXXCLANG19
// RUN: %clang_cc1 -triple thumbv7k-apple-watchos2.0 -target-abi aapcs16 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WATCHOS

// Empty structs are ignored for PCS purposes on WatchOS and in C mode
// elsewhere.  In C++ mode they consume a register slot though. Functions are
// slightly bigger than minimal to make confirmation against actual GCC
// behaviour easier.

#if __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

struct Empty {};

// C: define{{.*}} i32 @empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @empty_arg(i8 %e.coerce, i32 noundef %a)
// CXXCLANG19: define{{.*}} i32 @empty_arg(i32 noundef %a)
// WATCHOS: define{{.*}} i32 @empty_arg(i32 noundef %a)
EXTERNC int empty_arg(struct Empty e, int a) {
  return a;
}

// C: define{{.*}} void @empty_ret()
// CXX: define{{.*}} void @empty_ret()
// CXXCLANG19: define{{.*}} void @empty_ret()
// WATCHOS: define{{.*}} void @empty_ret()
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

// C: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
// CXXCLANG19: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
// WATCHOS: define{{.*}} i32 @super_empty_arg(i32 noundef %a)
EXTERNC int super_empty_arg(struct SuperEmpty e, int a) {
  return a;
}

struct SortOfEmpty {
  struct SuperEmpty e;
};

// C: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
// CXX: define{{.*}} i32 @sort_of_empty_arg(i8 %e.coerce, i32 noundef %a)
// CXXCLANG19: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
// WATCHOS: define{{.*}} i32 @sort_of_empty_arg(i32 noundef %a)
EXTERNC int sort_of_empty_arg(struct Empty e, int a) {
  return a;
}

// C: define{{.*}} void @sort_of_empty_ret()
// CXX: define{{.*}} void @sort_of_empty_ret()
// CXXCLANG19: define{{.*}} void @sort_of_empty_ret()
// WATCHOS: define{{.*}} void @sort_of_empty_ret()
EXTERNC struct SortOfEmpty sort_of_empty_ret(void) {
  struct SortOfEmpty e;
  return e;
}

#include <stdarg.h>

// va_arg matches the above rules, consuming an incoming argument in cases
// where one would be passed, and not doing so when the argument should be
// ignored.

EXTERNC int empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @empty_arg_variadic(
// C: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// C-NOT: {{ getelementptr }}
// CXX: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXX: %argp.next2 = getelementptr inbounds i8, ptr %argp.cur1, i32 4
// CXXCLANG19: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXXCLANG19-NOT: {{ getelementptr }}
// WATCHOS: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// WATCHOS-NOT: {{ getelementptr }}
  va_list vl;
  va_start(vl, a);
  struct Empty b = va_arg(vl, struct Empty);
  int c = va_arg(vl, int);
  va_end(vl);
  return c;
}

EXTERNC int super_empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @super_empty_arg_variadic(
// C: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// C-NOT: {{ getelementptr }}
// CXX: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXX-NOT: {{ getelementptr }}
// CXXCLANG19: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXXCLANG19-NOT: {{ getelementptr }}
// WATCHOS: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// WATCHOS-NOT: {{ getelementptr }}
  va_list vl;
  va_start(vl, a);
  struct SuperEmpty b = va_arg(vl, struct SuperEmpty);
  int c = va_arg(vl, int);
  va_end(vl);
  return c;
}

EXTERNC int sort_of_empty_arg_variadic(int a, ...) {
// CHECK-LABEL: @sort_of_empty_arg_variadic(
// C: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// C-NOT: {{ getelementptr }}
// CXX: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXX-NOT: {{ getelementptr }}
// CXXCLANG19: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// CXXCLANG19-NOT: {{ getelementptr }}
// WATCHOS: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
// WATCHOS-NOT: {{ getelementptr }}
  va_list vl;
  va_start(vl, a);
  struct SortOfEmpty b = va_arg(vl, struct SortOfEmpty);
  int c = va_arg(vl, int);
  va_end(vl);
  return c;
}
