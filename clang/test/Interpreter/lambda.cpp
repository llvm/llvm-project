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

// in clang-repl, allow lambda to capture top level variable
int x = 42;

// expected-no-diagnostics
auto capture = [&]() { return x * 2; };
auto capture2 = [&x]() { return x * 2; };

printf("capture = %d\n", capture());
// CHECK: capture = 84

printf("capture2 = %d\n", capture2());
// CHECK: capture2 = 84

// Ensure interpreter continues and x is still valid
printf("x = %d\n", x);
// CHECK: x = 42

%quit
