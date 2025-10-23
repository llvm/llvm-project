// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s
// At -O2, somehow "x = 42" appears first when piped into FileCheck,
// see https://github.com/llvm/llvm-project/issues/143547.
// RUN: %if !system-windows %{ cat %s | clang-repl -Xcc -Xclang -Xcc -verify -Xcc -O2 | FileCheck %s %}

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

// expected-error {{non-local lambda expression cannot have a capture-default}}
auto capture = [&]() { return x * 2; };

// Ensure interpreter continues and x is still valid
printf("x = %d\n", x);
// CHECK: x = 42

%quit
