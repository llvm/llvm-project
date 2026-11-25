// The temporaries of a printed expression are destroyed at the end of its
// statement, and their destructors are emitted.
//
// RUN: cat %s | clang-repl | FileCheck %s

// The test is flaky with ASan: https://github.com/llvm/llvm-project/issues/102858
// UNSUPPORTED: asan

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

// A printed object of class type is constructed in the storage of the value
// by placement new, but the temporaries of its initializer still need the
// cleanups.
struct T { int I; };
T h(S) { return T{5}; }

h(S())
// CHECK-NEXT: (T) @0x{{[0-9a-f]+}}

Dtors
// CHECK-NEXT: (int) 4

%quit
