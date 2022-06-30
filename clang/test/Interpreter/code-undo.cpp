// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// CHECK-DRIVER: i = 10
// RUN: cat %s | clang-repl | FileCheck %s
extern "C" int printf(const char *, ...);
int x1 = 0;
int x2 = 42;
%undo
int x2 = 24;
auto r1 = printf("x1 = %d\n", x1);
// CHECK: x1 = 0
auto r2 = printf("x2 = %d\n", x2);
// CHECK-NEXT: x2 = 24

int foo() { return 1; }
%undo
int foo() { return 2; }
auto r3 = printf("foo() = %d\n", foo());
// CHECK-NEXT: foo() = 2

%quit
