// Shared type descriptors survive the conversion of compiler-generated names:
// device code still loads the type descriptor address from the managed
// pointer when type descriptors are renamed for assembly.

// RUN: fir-opt --cuf-shared-type-info --compiler-generated-names --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s

module attributes {gpu.container_module} {
  fir.global linkonce_odr @_QMmE.dt.t constant : i8 {
    %0 = arith.constant 0 : i8
    fir.has_value %0 : i8
  }

  gpu.module @cuda_device_mod {
    fir.global linkonce_odr @_QMmE.dt.t constant : i8 {
      %0 = arith.constant 0 : i8
      fir.has_value %0 : i8
    }

    func.func @embox(%arg0: !fir.ref<!fir.type<_QMmTt{i:i32}>>) -> !fir.box<!fir.type<_QMmTt{i:i32}>> {
      %0 = fir.embox %arg0 : (!fir.ref<!fir.type<_QMmTt{i:i32}>>) -> !fir.box<!fir.type<_QMmTt{i:i32}>>
      return %0 : !fir.box<!fir.type<_QMmTt{i:i32}>>
    }

    func.func @type_desc() -> !fir.tdesc<!fir.type<_QMmTt{i:i32}>> {
      %0 = fir.type_desc !fir.type<_QMmTt{i:i32}>
      return %0 : !fir.tdesc<!fir.type<_QMmTt{i:i32}>>
    }

    func.func @address_of() -> !fir.ref<i8> {
      %0 = fir.address_of(@_QMmE.dt.t) : !fir.ref<i8>
      return %0 : !fir.ref<i8>
    }
  }
}

// CHECK: llvm.mlir.global linkonce_odr @_QMmEXdtXt()
// CHECK: llvm.mlir.global external @[[PTR:_QMmEXdtXtXhostaddr[0-9a-f]+]]()

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK-SAME: cuf.shared_type_descs = {_QMmEXdtXt = @[[PTR]]}
// CHECK: llvm.mlir.global linkonce_odr constant @_QMmEXdtXt()

// CHECK-LABEL: llvm.func @embox
// CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[PTR]] : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.insertvalue %[[TYPEDESC]], %{{.*}}[{{[0-9]+}}] : !llvm.struct<

// CHECK-LABEL: llvm.func @type_desc
// CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[PTR]] : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.return %[[TYPEDESC]] : !llvm.ptr

// CHECK-LABEL: llvm.func @address_of
// CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[PTR]] : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.return %[[TYPEDESC]] : !llvm.ptr

// CHECK: llvm.mlir.global external @[[PTR]]() {{.*}}nvvm.managed
