// RUN: %clang_cc1 -std=c89 -emit-llvm %s -o - | FileCheck %s

// There should not be an unresolved reference to func here.  Believe it or not,
// the "expected result" is a function named 'func' which is internal and
// referenced by bar().

// This is PR244

// CHECK-LABEL: define {{.*}}void @bar(
// CHECK: call {{.*}} @func
// CHECK: define internal {{.*}}i32 @func(
static int func();
void bar(void) {
  int func(void);
  foo(func);
}
static int func(char** A, char ** B) { return 0; }
