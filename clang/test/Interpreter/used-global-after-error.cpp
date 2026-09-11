// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl | FileCheck %s

// A cast<> of a null WeakTrackingVH used to crash emitUsed when a global on
// the llvm.used list was deleted before the module was released: the
// duplicate definition below is diagnosed by CodeGen, which then replaces the
// used global's unreferenced GV with a differently-typed one; the failed
// parse keeps the module alive, and the next successful parse finalizes it.
extern "C" int printf(const char *, ...);
__attribute__((used)) int a asm("sym") = 1; float b asm("sym") = 2.0f;
auto r1 = printf("ok = %d\n", 42);
// CHECK: ok = 42

%quit
