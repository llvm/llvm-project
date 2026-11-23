// Test that dso_local is not set on non-ELF targets, even with static
// relocation model. Uses a separate file because fir-to-llvm-ir target=
// overrides the module triple for all splits.

// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-apple-darwin" %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-apple-darwin"} {
  fir.global @macho_def : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @macho_def()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32
