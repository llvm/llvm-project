// RUN: %clang_cc1 -triple=i386-pc-solaris2.11 -w -emit-llvm %s -o - | FileCheck %s

extern "C" {
  struct statvfs64 {
    int f;
  };
#pragma redefine_extname statvfs64 statvfs
  int statvfs64(struct statvfs64 *);
}

void some_func() {
  struct statvfs64 st;
  statvfs64(&st);
// Check that even if there is a structure with redefined name before the
// pragma, subsequent function name redefined properly. PR5172, Comment 11.
// CHECK:  call i32 @statvfs(ptr noundef %st)
}

// This is a case when redefenition is deferred *and* we have a local of the
// same name. PR23923.
#pragma redefine_extname foo bar
int f() {
  int foo = 0;
  return foo;
}
extern "C" {
  int foo() { return 1; }
// CHECK: define{{.*}} i32 @bar()
}

// Check that #pragma redefine_extname applies to C code only, and shouldn't be
// applied to C++.
#pragma redefine_extname foo_cpp bar_cpp
extern int foo_cpp() { return 1; }
// CHECK-NOT: define{{.*}} i32 @bar_cpp()

