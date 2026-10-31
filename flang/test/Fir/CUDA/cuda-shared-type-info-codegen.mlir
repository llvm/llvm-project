// Under -gpu=mem:managed, device code loads the address of a shared type
// descriptor from the managed pointer recorded in the GPU module attribute
// cuf.shared_type_descs (see cuda-shared-type-info.mlir).

// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod attributes {cuf.shared_type_descs = {_QMmEXdtXt = @_QMmEXdtXtXhostaddr0}} {
    fir.global linkonce_odr @_QMmE.dt.t constant : i8
    fir.global @_QMmEXdtXtXhostaddr0 <{data_attr = #cuf.cuda<managed>}> : !fir.llvm_ptr<i8> {
      %0 = fir.zero_bits !fir.llvm_ptr<i8>
      fir.has_value %0 : !fir.llvm_ptr<i8>
    }
    // An initializer needs a constant address and keeps the device copy.
    fir.global @_QMmEholder constant : !fir.llvm_ptr<i8> {
      %0 = fir.address_of(@_QMmE.dt.t) : !fir.ref<i8>
      %1 = fir.convert %0 : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
      fir.has_value %1 : !fir.llvm_ptr<i8>
    }
    fir.global @_QMmEboxed : !fir.box<!fir.ptr<!fir.type<_QMmTt{i:i32}>>> {
      %0 = fir.zero_bits !fir.ptr<!fir.type<_QMmTt{i:i32}>>
      %1 = fir.embox %0 : (!fir.ptr<!fir.type<_QMmTt{i:i32}>>) -> !fir.box<!fir.ptr<!fir.type<_QMmTt{i:i32}>>>
      fir.has_value %1 : !fir.box<!fir.ptr<!fir.type<_QMmTt{i:i32}>>>
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

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: llvm.mlir.global external @_QMmEXdtXtXhostaddr0() {{.*}}nvvm.managed

// CHECK: llvm.mlir.global {{.*}}@_QMmEholder()
// CHECK: llvm.mlir.addressof @_QMmE.dt.t : !llvm.ptr

// CHECK: llvm.mlir.global {{.*}}@_QMmEboxed()
// CHECK-NOT: llvm.load
// CHECK: llvm.mlir.addressof @_QMmE.dt.t : !llvm.ptr
// CHECK-NOT: llvm.load
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @embox
// CHECK: %[[PTR:.*]] = llvm.mlir.addressof @_QMmEXdtXtXhostaddr0 : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.insertvalue %[[TYPEDESC]], %{{.*}}[{{[0-9]+}}] : !llvm.struct<

// CHECK-LABEL: llvm.func @type_desc
// CHECK: %[[PTR:.*]] = llvm.mlir.addressof @_QMmEXdtXtXhostaddr0 : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.return %[[TYPEDESC]] : !llvm.ptr

// CHECK-LABEL: llvm.func @address_of
// CHECK: %[[PTR:.*]] = llvm.mlir.addressof @_QMmEXdtXtXhostaddr0 : !llvm.ptr<1>
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[TYPEDESC:.*]] = llvm.load %[[CAST]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.return %[[TYPEDESC]] : !llvm.ptr
