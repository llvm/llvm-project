// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
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
