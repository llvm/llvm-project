// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc  -verify | FileCheck %s
// RUN: %clang_cc1 -verify -fincremental-extensions -emit-llvm -o -  %s \
// RUN:           | FileCheck --check-prefix=CODEGEN-CHECK %s

// expected-no-diagnostics

//CODEGEN-CHECK-COUNT-2: define internal void @__stmts__
//CODEGEN-CHECK-NOT: define internal void @__stmts__

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

{i = 0; printf("i = %d (global scope)\n", i);}
// CHECK-NEXT: i = 0

while (int i = 1) { printf("i = %d (while condition)\n", i--); break; }
// CHECK-NEXT: i = 1

if (int i = 2) printf("i = %d (if condition)\n", i);
// CHECK-NEXT: i = 2

switch (int i = 3) { default: printf("i = %d (switch condition)\n", i); }
// CHECK-NEXT: i = 3

for (int i = 4; i > 3; --i) printf("i = %d (for-init)\n", i);
// CHECK-NEXT: i = 4

for (const auto &i : "5") printf("i = %c (range-based for-init)\n", i);
// CHECK-NEXT: i = 5

int *aa=nullptr;
if (auto *b=aa) *b += 1;
while (auto *b=aa) ;
for (auto *b=aa; b; *b+=1) ;
