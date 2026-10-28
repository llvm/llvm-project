// The global ending the shared type information is appended to an existing
// llvm.used array.

// RUN: fir-opt --cuf-shared-type-info %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global linkonce_odr @_QMmE.n.t constant : !fir.char<1> {
    %0 = fir.string_lit "t"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
  llvm.mlir.global internal @keep(0 : i32) : i32
  llvm.mlir.global appending @llvm.used() {section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.addressof @keep : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr>
    llvm.return %2 : !llvm.array<1 x ptr>
  }
}

// CHECK-NOT: @llvm.used
// CHECK: llvm.mlir.global appending @llvm.used() {{.*}}: !llvm.array<2 x ptr> {
// CHECK: %[[KEEP:.*]] = llvm.mlir.addressof @keep : !llvm.ptr
// CHECK: %[[OLD0:.*]] = llvm.insertvalue %[[KEEP]], %{{.*}}[0] : !llvm.array<1 x ptr>
// CHECK: %[[NEW:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr>
// CHECK: %[[ELT:.*]] = llvm.extractvalue %[[OLD0]][0] : !llvm.array<1 x ptr>
// CHECK: %[[NEW0:.*]] = llvm.insertvalue %[[ELT]], %[[NEW]][0] : !llvm.array<2 x ptr>
// CHECK: %[[MARKER:.*]] = llvm.mlir.addressof @__nv_type_info_end_{{[0-9a-f]+}} : !llvm.ptr
// CHECK: %[[NEW1:.*]] = llvm.insertvalue %[[MARKER]], %[[NEW0]][1] : !llvm.array<2 x ptr>
// CHECK: llvm.return %[[NEW1]] : !llvm.array<2 x ptr>
// CHECK: llvm.mlir.global internal @__nv_type_info_end_{{[0-9a-f]+}}(0 : i8)
// CHECK-NOT: @llvm.used
