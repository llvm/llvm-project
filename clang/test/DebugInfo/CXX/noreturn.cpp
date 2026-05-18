// RUN: %clang_cc1 -std=c++11 -emit-llvm -fcxx-exceptions -debug-info-kind=standalone  %s -o - | FileCheck %s
// Test for NoReturn flags in debug info.

// CHECK: DISubprogram(name: "f", {{.*}}, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagDefinition
// CHECK: DISubprogram(name: "foo_member", {{.*}}, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0
// CHECK-NOT: DISubprogram(name: "func",{{.*}}, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagDefinition

class foo {

  [[noreturn]] void foo_member() { throw 1; }
};

[[noreturn]] void f() {
  throw 1;
}

void func() {
  foo object;
}
