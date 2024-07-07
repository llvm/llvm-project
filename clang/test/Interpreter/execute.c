// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix

// RUN: cat %s | clang-repl -Xcc -xc -Xcc -Xclang -Xcc -verify | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -xc -Xcc -O2 -Xcc -Xclang -Xcc -verify| FileCheck %s
int printf(const char *, ...);
int i = 42; err // expected-error{{use of undeclared identifier}}
int i = 42;
struct S { float f; struct S *m;} s = {1.0, 0};
// FIXME: Making foo inline fails to emit the function.
int foo() { return 42; }
void run() {                                                    \
  printf("i = %d\n", i);                                        \
  printf("S[f=%f, m=0x%llx]\n", s.f, (unsigned long long)s.m);  \
  int r3 = foo();                                               \
}
run();
// CHECK: i = 42
// CHECK-NEXT: S[f=1.000000, m=0x0]

%quit
