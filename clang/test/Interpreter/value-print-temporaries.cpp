// The temporaries of a printed expression are destroyed at the end of its
// statement, and their destructors are emitted.
//
// RUN: cat %s | clang-repl | FileCheck %s

int Dtors = 0;
struct S { ~S() { ++Dtors; } };
int f(S) { return 42; }

f(S())
// CHECK: (int) 42

Dtors
// CHECK-NEXT: (int) 1

struct R { int I; ~R() { ++Dtors; } };
R g(const S &) { return R{7}; }

g(S()).I
// CHECK-NEXT: (int) 7

Dtors
// CHECK-NEXT: (int) 3

%quit
