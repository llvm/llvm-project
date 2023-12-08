// UNSUPPORTED: system-aix
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

inline int bar() { return 42;}
auto r4 = bar();
%undo
auto r5 = bar();

%quit
