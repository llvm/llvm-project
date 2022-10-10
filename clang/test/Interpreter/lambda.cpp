// RUN: clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"
// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// CHECK-DRIVER: i = 10
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -O2 | FileCheck %s
extern "C" int printf(const char *, ...);

auto l1 = []() { printf("ONE\n"); return 42; };
auto l2 = []() { printf("TWO\n"); return 17; };

auto r1 = l1();
// CHECK: ONE
auto r2 = l2();
// CHECK: TWO
auto r3 = l2();
// CHECK: TWO

%quit
