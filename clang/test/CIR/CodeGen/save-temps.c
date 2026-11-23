// Test that -save-temps with -fclangir emits .cir and .mlir intermediate files.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -save-temps=obj -o %t.ll %s

// Check that the .cir file was created and contains CIR dialect ops
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// CIR: cir.func
// CIR: cir.return

// Check that the .mlir file was created and contains LLVM dialect ops
// RUN: FileCheck --input-file=%t.mlir %s --check-prefix=MLIR

// MLIR: llvm.func
// MLIR: llvm.return

int foo(int x) {
  return x + 1;
}
