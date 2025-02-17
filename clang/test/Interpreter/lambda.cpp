// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | not clang-repl -Xcc -Xclang -Xcc -verify -Xcc -O2 | FileCheck %s

extern "C" int printf(const char *, ...);

auto l1 = []() { printf("ONE\n"); return 42; };
auto l2 = []() { printf("TWO\n"); return 17; };

auto r1 = l1();
// CHECK: ONE
auto r2 = l2();
// CHECK: TWO
auto r3 = l2();
// CHECK: TWO

// Verify non-local lambda capture error is correctly reported
int x = 42;

// expected-error@+1 {{non-local lambda expression cannot have a capture-default}}
auto capture = [&]() { return x * 2; };

%quit