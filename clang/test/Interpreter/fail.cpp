// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// clang-repl can be called from the prompt in non-interactive mode as a
// calculator in shell scripts, for example. In that case if there is an error
// we should set the exit code as failure.
// RUN: not clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"

// In interactive (REPL) mode, we can have errors but we should exit with
// success because errors in the input code are part of the interactive use.
// RUN: cat %s | clang-repl | FileCheck %s

// However, interactive mode should fail when we specified -verify and there
// was a diagnostic mismatches. This will make the testsuite fail as intended.
// RUN: cat %s | not clang-repl -Xcc -Xclang -Xcc -verify | FileCheck %s

BOOM! // expected-error {{intended to fail the -verify test}}
extern "C" int printf(const char *, ...);
int i = 42;
auto r1 = printf("i = %d\n", i);
// CHECK: i = 42
%quit
