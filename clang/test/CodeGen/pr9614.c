// RUN: %clang_cc1 -emit-llvm %s -O1 -o - | FileCheck %s

extern void foo_alias (void) __asm ("foo");
inline void foo (void) {
  return foo_alias ();
}
extern void bar_alias (void) __asm ("bar");
inline __attribute__ ((__always_inline__)) void bar (void) {
  return bar_alias ();
}
void f(void) {
  foo();
  bar();
}

// CHECK: define void @f()
// CHECK: call void @foo()
// CHECK-NEXT: call void @bar()
// CHECK-NEXT: ret void

// CHECK: declare void @foo()
// CHECK: declare void @bar()
