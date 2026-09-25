// RUN: not %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm -x cuda \
// RUN:     -target-sdk-version=12.3 -fcuda-include-gpubinary %t.missing.bin \
// RUN:     %s -o %t.ll 2>&1 | FileCheck %s

// LoweringPrepare emits an MLIR-side error via mlir::Operation::emitError when
// the requested CUDA gpubinary cannot be opened. With CIRDiagnosticHandler
// installed, that diagnostic surfaces through clang's DiagnosticsEngine in
// clang's standard format (`error: ...`) rather than MLIR's
// `loc("file":N:M): error: ...` default-handler format.

// CHECK: error: cannot open GPU binary file: {{.*}}.missing.bin
// CHECK-NOT: loc({{.*}}): error: cannot open GPU binary file

// The generic CIR-to-CIR transform fatal error must NOT be reported on top of
// the specific MLIR-relayed error. CIRGenAction gates the fallback diag on
// clang::DiagnosticsEngine::hasErrorOccurred() so users see one root cause,
// not two.
// CHECK-NOT: error: CIR-to-CIR transformation failed

__attribute__((global)) void kernel() {}
