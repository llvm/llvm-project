// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

int __cdecl myexit(void (__cdecl *pf)(void));

struct S {
  S();
  ~S();
};

#pragma init_seg(".myseg", myexit)

S s;

// The initializer pointer is still placed in the custom section.
// CHECK: @__cxx_init_fn_ptr = private constant ptr @"??__Es@@YAXXZ", section ".myseg"

// The destructor registration calls myexit instead of atexit.
// CHECK-LABEL: define {{.*}} @"??__Es@@YAXXZ"
// CHECK: call i32 @"?myexit@@{{[^"]+}}"(
// CHECK-NOT: call {{.*}} @atexit(
