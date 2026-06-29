// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Out-of-line member + `if` + EH cleanups + early `return` used to emit a
// `cir.if` "then" region whose block ended on `cir.cleanup.scope` with no
// following `cir.yield`, tripping MLIR's "block with no terminator" verifier.

struct S {
  ~S();
};

int f(bool b) {
  if (b) {
    S temp;
    return 1;
  }
  return 0;
}

// CHECK-LABEL: cir.func{{.*}} @_Z1fb
// CHECK: cir.if
// CHECK: cir.cleanup.scope
// CHECK: cir.yield
