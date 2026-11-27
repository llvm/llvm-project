// UNSUPPORTED: system-windows
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);
int __attribute__((weak)) bar() { return 42; }
auto r4 = printf("bar() = %d\n", bar());
// CHECK: bar() = 42

int a = 12;
static __typeof(a) b __attribute__((__weakref__("a")));
int c = b;

%quit
