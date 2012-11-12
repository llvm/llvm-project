// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck %s

extern void foo_alias (void) __asm ("foo");
inline void foo (void) {
  return foo_alias ();
}
extern void bar_alias (void) __asm ("bar");
inline __attribute__ ((__always_inline__)) void bar (void) {
  return bar_alias ();
}
extern char *strrchr_foo (const char *__s, int __c)  __asm ("strrchr");
extern inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) char * strrchr_foo (const char *__s, int __c)  {
  return __builtin_strrchr (__s, __c);
}
void f(void) {
  foo();
  bar();
  strrchr_foo("", '.');
}

// CHECK: define void @f()
// CHECK: call void @foo()
// CHECK-NEXT: call void @bar()
// CHECK-NEXT: call i8* @strrchr(
// CHECK-NEXT: ret void

// CHECK: declare void @foo()
// CHECK: declare void @bar()
// CHECK: declare i8* @strrchr(i8*, i32)
