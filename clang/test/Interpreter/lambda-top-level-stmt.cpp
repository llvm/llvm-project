// Lambdas in top-level statements used to crash the Itanium mangler, which
// cast their TopLevelStmtDecl context to NamedDecl. Two lambdas verify that
// the closure types still mangle to distinct names.
// REQUIRES: host-supports-jit
// MSVC compat enables -fdelayed-template-parsing, which hits a pre-existing
// Sema::PushDeclContext assert on any late-parsed template instantiation in
// incremental mode, independent of this fix.
// UNSUPPORTED: system-windows
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);

namespace ns { template <typename F> void call(F f) { f(); } }

ns::call([] { printf("ONE\n"); });
// CHECK: ONE
ns::call([] { printf("TWO\n"); });
// CHECK-NEXT: TWO

%quit
