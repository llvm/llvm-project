// FIXME: There're some inconsistencies between interactive and non-interactive
// modes. For example, when clang-repl runs in the interactive mode, issues an
// error, and then successfully recovers if we decide it's a success then for
// the non-interactive mode the exit code should be a failure.
// RUN: clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"
// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | not clang-repl | FileCheck %s
BOOM!
extern "C" int printf(const char *, ...);
int i = 42;
auto r1 = printf("i = %d\n", i);
// CHECK: i = 42
%quit
