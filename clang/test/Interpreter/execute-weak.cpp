// RUN: clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"
// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// CHECK-DRIVER: i = 10
//
// UNSUPPORTED: system-aix, system-windows
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);
int __attribute__((weak)) bar() { return 42; }
auto r4 = printf("bar() = %d\n", bar());
// CHECK: bar() = 42

int a = 12;
static __typeof(a) b __attribute__((__weakref__("a")));
int c = b;

%quit
