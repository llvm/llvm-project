// RUN: not %clang_cc1 -triple x86_64-apple-macosx10.15 -fclangir -emit-llvm \
// RUN:     %s -o %t.ll 2>&1 | FileCheck %s

// LoweringPrepare emits an MLIR-side error via mlir::Operation::emitError for
// a thread_local variable on a target whose thread wrapper is replaceable.
// CIRDiagnosticHandler must surface it in clang's `file:line:col: error: ...`
// format rather than MLIR's `loc("file":N:M): error: ...` default.

// CHECK: mlir-error-routing.cpp:[[#@LINE+7]]:1: error: Unhandled thread wrapper attributes for CC and Nounwind
// CHECK-NOT: loc({{.*}}): error: Unhandled thread wrapper attributes

struct S {
  S();
  ~S();
};
thread_local S s;
S *use() { return &s; }

// The generic CIR-to-CIR transform fatal error must not be reported on top of
// the specific one: CIRGenAction gates it on hasErrorOccurred().
// CHECK-NOT: error: CIR-to-CIR transformation failed
