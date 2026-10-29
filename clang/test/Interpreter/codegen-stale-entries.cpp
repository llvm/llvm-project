// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
//
// This test demonstrates that stale entries in CodeGen keep foo and infn
// reachable after %undo. It motivates removing stale entries from
// CodeGenModule::DeferredDecls and CodeGenModule::Manglings via
// CodeGenModule::forgetDecl (PR #234493).

extern "C" int printf(const char *, ...);

static int foo(int x);
static int foo(int x) { return x + 1; }
%undo
auto sfr = printf("foo=%d\n", foo(10));
// CHECK-NOT: foo=11
// CHECK: JIT session error: Symbols not found

inline int infn(int x);
inline int infn(int x) { return x + 1; }
%undo
auto infnr = printf("infn=%d\n", infn(10));
// CHECK-NOT: infn=11
// CHECK: JIT session error: Symbols not found

%quit
