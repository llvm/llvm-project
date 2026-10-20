// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-directives-only %s -o - | FileCheck %s

// Inserting lifetime markers should not affect debuginfo: lifetime.end is not
// a destructor, but instrumentation for the compiler. Ensure the debug info for
// the return statement (in the IR) does not point to the function closing '}'
// which is used to show some destructors have been called before leaving the
// function.

extern int f(int);
extern int g(int);

// CHECK-LABEL: define{{.*}} i32 @test
int test(int a, int b) {
  int res;

  if (a==2) {
    int r = f(b);
    res = r + b;
    a += 2;
  } else {
    int r = f(a);
    res = r + a;
    b += 1;
  }

  return res;
// CHECK: ret i32 %{{.*}}, !dbg [[DI:![0-9]+]]
// CHECK: [[DI]] = !DILocation(line: [[@LINE-2]]
}
