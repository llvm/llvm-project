// clang-format off
// UNSUPPORTED: system-aix
// XFAIL for arm, or running on Windows.
// XFAIL: target=arm-{{.*}}, target=armv{{.*}}, system-windows
// RUN: cat %s | clang-repl | FileCheck %s

// Incompatible with msan. It passes with -O3 but fail -Oz. Interpreter
// generates non-instrumented code, which may call back to instrumented.
// UNSUPPORTED: msan

extern "C" int printf(const char *, ...);

int f() { throw "Simple exception"; return 0; }
int checkException() { try { printf("Running f()\n"); f(); } catch (const char *e) { printf("%s\n", e); } return 0; }
auto r1 = checkException();
// CHECK: Running f()
// CHECK-NEXT: Simple exception

%quit
