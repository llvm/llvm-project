// RUN: clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"
// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// CHECK-DRIVER: i = 10
// RUN: cat %s | clang-repl | FileCheck %s
extern "C" int printf(const char *, ...);
int i = 42;
auto r1 = printf("i = %d\n", i);
// CHECK: i = 42

struct S { float f = 1.0; S *m = nullptr;} s;

auto r2 = printf("S[f=%f, m=0x%llx]\n", s.f, reinterpret_cast<unsigned long long>(s.m));
// CHECK-NEXT: S[f=1.000000, m=0x0]

inline int foo() { return 42; }
int r3 = foo();

int __attribute__((weak)) bar() { return 1; }
auto r4 = printf("bar() = %d\n", bar());
// CHECK-NEXT: bar() = 1

%quit
