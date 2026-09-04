// RUN: inter-opt %s --inter-import-llvm | FileCheck %s
// RUN: inter-opt %s '--inter-import-llvm=simd-width=8' | FileCheck %s --check-prefix=WIDTH8

module {
  llvm.func spir_kernelcc @branch_args(%condition: i1, %value: i32) {
    llvm.cond_br %condition, ^then, ^merge(%value : i32)
  ^then:
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^merge(%one : i32)
  ^merge(%result: i32):
    llvm.return
  }

  llvm.func spir_kernelcc @abi(%byte: i8, %word: i32, %wide: i64,
                               %single: f32,
                               %input: !llvm.ptr<1> {llvm.readonly},
                               %output: !llvm.ptr<1> {llvm.writeonly}) {
    llvm.return
  }

  llvm.func spir_kernelcc @payload_boundary(%word: i32, %wide: i64) {
    llvm.return
  }
}

// CHECK-LABEL: func.func @branch_args
// CHECK-SAME: attributes {
// CHECK-SAME: xw.kernel
// CHECK: cf.cond_br {{.*}}, ^bb1, ^bb2({{.*}} : i32)
// CHECK: cf.br ^bb2({{.*}} : i32)
// CHECK: return

// CHECK-LABEL: func.func @abi
// CHECK-SAME: xw.kernel_args = [
// CHECK-SAME: {alignment = 1 : i64, kind = "value", offset = 24 : i64, size = 1 : i64}
// CHECK-SAME: {alignment = 4 : i64, kind = "value", offset = 28 : i64, size = 4 : i64}
// CHECK-SAME: {alignment = 4 : i64, kind = "value", offset = 32 : i64, size = 8 : i64}
// CHECK-SAME: {alignment = 4 : i64, kind = "value", offset = 40 : i64, size = 4 : i64}
// CHECK-SAME: {access = "read_only", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 48 : i64, size = 8 : i64}
// CHECK-SAME: {access = "write_only", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 56 : i64, size = 8 : i64}
// CHECK-SAME: xw.simd_width = 16 : i32
// CHECK-NOT: llvm.readonly
// CHECK-NOT: llvm.writeonly

// CHECK-LABEL: func.func @payload_boundary
// CHECK-SAME: {alignment = 4 : i64, kind = "value", offset = 24 : i64, size = 4 : i64}
// CHECK-SAME: {alignment = 4 : i64, kind = "value", offset = 32 : i64, size = 8 : i64}

// WIDTH8-COUNT-3: xw.simd_width = 8 : i32
