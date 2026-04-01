// Test that -save-temps with -fclangir emits .cir and .aiir intermediate files.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -save-temps=obj -o %t.ll %s

// Check that the .cir file was created and contains CIR dialect ops
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// CIR: cir.func
// CIR: cir.return

// Check that the .aiir file was created and contains LLVM dialect ops
// RUN: FileCheck --input-file=%t.aiir %s --check-prefix=AIIR

// AIIR: llvm.func
// AIIR: llvm.return

int foo(int x) {
  return x + 1;
}
