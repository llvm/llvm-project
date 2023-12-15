// RUN: clang-repl "int x = 10;" "{ int t; a::b(t); }" "int y = 10;"
// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | not clang-repl | FileCheck %s
{ int t; a::b(t); }
extern "C" int printf(const char *, ...);
int i = 42;
auto r1 = printf("i = %d\n", i);
// CHECK: i = 42
%quit
