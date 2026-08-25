// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl > %t.out 2>&1
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --check-prefix=NEGATIVE --input-file=%t.out

// An input that declares something with C language linkage and then fails must
// not take the interpreter down with it.

extern "C" int printf(const char *, ...);

// An error in the body of an extern "C" function definition.
extern "C" void f1() { undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
printf("alive %d\n", 1);
// CHECK-DAG: alive 1

// The same, written as an `extern "C" { ... }` block.
extern "C" { void f2() { undeclared_thing; } }
printf("alive %d\n", 2);
// CHECK-DAG: alive 2

// An extern "C" *variable* whose initializer fails: variables are registered
// with the ExternCContext by a different Sema path than functions.
extern "C" int v1 = undeclared_thing;
printf("alive %d\n", 3);
// CHECK-DAG: alive 3

// A deleted destructor reached through a wrapper -- the shape CppInterOp's
// generated destructor wrappers hit.
class D { public: ~D() = delete; };
extern "C" void g(D *p) { delete p; }
// CHECK-DAG: error: attempt to use a deleted function
printf("alive %d\n", 4);
// CHECK-DAG: alive 4

// A block-scope `extern` inside an extern "C" function is registered with the
// ExternCContext but is already off the IdResolver by the time CleanUpPTU runs,
// because its own scope popped while the input was still being parsed. Removing
// it again therefore does not just fail to find it: with assertions off it
// clears the identifier's chain out from under whatever else is on it.
extern "C" void h1() { extern int fresh; undeclared_thing; } int fresh = 1;
printf("alive %d\n", 5);
// CHECK-DAG: alive 5

// Surviving is not enough: the discarded PTU must leave nothing behind, so the
// very same names have to be definable afterwards and the definitions have to
// be the ones that run.
extern "C" void f1() { printf("f1 ran\n"); }
extern "C" void f2() { printf("f2 ran\n"); }
extern "C" int v1 = 5;
f1();
f2();
printf("v1 %d\n", v1);
// CHECK-DAG: f1 ran
// CHECK-DAG: f2 ran
// CHECK-DAG: v1 5

// Nothing anywhere in the session may claim the recovered definitions clash
// with what the discarded inputs left behind.
// NEGATIVE-NOT: error: redefinition
// NEGATIVE-NOT: error: conflicting types

%quit
