// Symbol references are resolved through the conversion's symbol tables.
// Check that a symbol is found both before and after its definition, i.e.
// whether the referenced global is still a fir.global or has already been
// converted, in the module and inside a gpu.module, and that linkonce
// globals keep their names when their comdat selectors are added.
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

func.func @use_before() -> !fir.ref<i32> {
  %0 = fir.address_of(@glob) : !fir.ref<i32>
  return %0 : !fir.ref<i32>
}

fir.global linkonce_odr @glob : i32 {
  %0 = arith.constant 1 : i32
  fir.has_value %0 : i32
}

func.func @use_after() -> !fir.ref<i32> {
  %0 = fir.address_of(@glob) : !fir.ref<i32>
  return %0 : !fir.ref<i32>
}

fir.global linkonce_odr @glob2 constant : i32 {
  %0 = arith.constant 2 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: llvm.func @use_before()
// CHECK:         llvm.mlir.addressof @glob : !llvm.ptr
// CHECK:       llvm.mlir.global linkonce_odr @glob() comdat(@__llvm_comdat::@glob) {addr_space = 0 : i32} : i32
// CHECK:       llvm.comdat @__llvm_comdat {
// CHECK-NEXT:    llvm.comdat_selector @glob any
// CHECK-NEXT:    llvm.comdat_selector @glob2 any
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func @use_after()
// CHECK:         llvm.mlir.addressof @glob : !llvm.ptr
// CHECK:       llvm.mlir.global linkonce_odr constant @glob2() comdat(@__llvm_comdat::@glob2) {addr_space = 0 : i32} : i32

// -----

// Inside a gpu.module the address space of a constant global comes from its
// cuf data attribute while it is a fir.global and from the llvm.mlir.global
// once converted; both references must get the same cast.

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @kernel_before(%arg0: !fir.ref<i32>) kernel {
      %0 = fir.address_of(@cst) : !fir.ref<i32>
      %1 = fir.load %0 : !fir.ref<i32>
      fir.store %1 to %arg0 : !fir.ref<i32>
      gpu.return
    }
    fir.global @cst {data_attr = #cuf.cuda<constant>} : i32 {
      %0 = fir.zero_bits i32
      fir.has_value %0 : i32
    }
    gpu.func @kernel_after(%arg0: !fir.ref<i32>) kernel {
      %0 = fir.address_of(@cst) : !fir.ref<i32>
      %1 = fir.load %0 : !fir.ref<i32>
      fir.store %1 to %arg0 : !fir.ref<i32>
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.func @kernel_before
// CHECK:         %[[A0:.*]] = llvm.mlir.addressof @cst : !llvm.ptr<4>
// CHECK:         llvm.addrspacecast %[[A0]] : !llvm.ptr<4> to !llvm.ptr
// CHECK:       llvm.mlir.global external @cst() {addr_space = 4 : i32} : i32
// CHECK-LABEL: gpu.func @kernel_after
// CHECK:         %[[A1:.*]] = llvm.mlir.addressof @cst : !llvm.ptr<4>
// CHECK:         llvm.addrspacecast %[[A1]] : !llvm.ptr<4> to !llvm.ptr
