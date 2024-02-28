// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc  -verify | FileCheck %s
// RUN: %clang_cc1 -verify -fincremental-extensions -emit-llvm -o -  %s \
// RUN:           | FileCheck --check-prefix=CODEGEN-CHECK %s

// expected-no-diagnostics

//CODEGEN-CHECK-COUNT-2: define internal void @__stmts__
//CODEGEN-CHECK-NOT: define internal void @__stmts__

// New tests fail right now
// XFAIL: *

extern "C" int printf(const char*,...);

template <typename T> T call() { printf("called\n"); return T(); }
call<int>();
// CHECK: called

int i = 1;
++i;
printf("i = %d\n", i);
// CHECK: i = 2

namespace Ns { void f(){ i++; } }
Ns::f();

void g() { ++i; }
g();
::g();

printf("i = %d\n", i);
// CHECK-NEXT: i = 5

for (; i > 4; --i) printf("i = %d\n", i);
// CHECK-NEXT: i = 5

{++i;}

for (; i > 4; --i) { printf("i = %d\n", i); };
// CHECK-NEXT: i = 5

int j = i; printf("j = %d\n", j);
// CHECK-NEXT: j = 4

if (int i = j) printf("i = %d\n", i);
// CHECK-NEXT: i = 4

for (int i = j; i > 3; --i) printf("i = %d\n", i);
// CHECK-NEXT: i = 4
