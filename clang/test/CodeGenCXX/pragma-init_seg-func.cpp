// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s --check-prefix=NOTCHECK

int __cdecl myexit(void (__cdecl *pf)(void));

struct S {
  S();
  ~S();
};

#pragma init_seg(".myseg", myexit)

S s;

// CHECK: @__cxx_init_fn_ptr = private constant ptr @"??__Es@@YAXXZ", section ".myseg"
// CHECK-LABEL: define {{.*}} @"??__Es@@YAXXZ"
// CHECK: call i32 @"?myexit
// CHECK: ret void

// NOTCHECK-LABEL: define {{.*}} @"??__Es@@YAXXZ"
// NOTCHECK-NOT: call {{.*}} @atexit
// NOTCHECK: ret void
