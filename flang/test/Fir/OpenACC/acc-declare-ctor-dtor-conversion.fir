// RUN: fir-opt %s -acc-declare-ctor-dtor-conversion -o - | FileCheck %s
// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-declare-ctor-dtor-conversion{generate-dtors=false})" -o - | FileCheck %s --check-prefix=NODTOR

// Derived from:
// module mm
// integer, parameter :: N = 7
// real :: arr(N)
// !$acc declare create(arr)
// end module

// CHECK-LABEL: module
// CHECK-NOT: acc.global_dtor
// CHECK-NOT: acc.global_ctor

// LLVM functions are created from acc constructors and destructors, then
// registered once in llvm.mlir.global_ctors / global_dtors (merged with any
// pre-existing entries such as the CUDA Fortran constructor).
// CHECK: llvm.func internal @_QMmmEarr_acc_ctor()
// CHECK: llvm.func internal @_QMmmEotherEarr_acc_ctor()
// CHECK: llvm.func internal @_QMmmEarr_acc_dtor()
// CHECK: llvm.func internal @_QMmmEotherEarr_acc_dtor()
// CHECK: llvm.mlir.global_ctors ctors = [{{.*}}@_QMmmEarr_acc_ctor{{.*}}@_QMmmEotherEarr_acc_ctor], priorities = [{{.*}}102 : i32{{.*}}102 : i32], data = [{{.*}}#llvm.zero{{.*}}#llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@_QMmmEarr_acc_dtor, @_QMmmEotherEarr_acc_dtor], priorities = [102 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]

// NODTOR-LABEL: module
// NODTOR-NOT: @_QMmmEarr_acc_dtor
// NODTOR-NOT: @_QMmmEotherEarr_acc_dtor
// NODTOR: llvm.func internal @_QMmmEarr_acc_ctor()
// NODTOR: llvm.func internal @_QMmmEotherEarr_acc_ctor()
// NODTOR: llvm.mlir.global_ctors ctors = [{{.*}}@_QMmmEarr_acc_ctor{{.*}}@_QMmmEotherEarr_acc_ctor], priorities = [{{.*}}102 : i32{{.*}}102 : i32], data = [{{.*}}#llvm.zero{{.*}}#llvm.zero]
// NODTOR-NOT: @_QMmmEarr_acc_dtor
// NODTOR-NOT: @_QMmmEotherEarr_acc_dtor
// NODTOR-NOT: llvm.mlir.global_dtors

module attributes {gpu.container_module} {
  fir.global @_QMmmEarr {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.array<7xf32> {
    %0 = fir.zero_bits !fir.array<7xf32>
    fir.has_value %0 : !fir.array<7xf32>
  }
  fir.global @_QMmmEotherEarr {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.array<3xf32> {
    %0 = fir.zero_bits !fir.array<3xf32>
    fir.has_value %0 : !fir.array<3xf32>
  }
  acc.global_ctor @_QMmmEarr_acc_ctor {
    %0 = fir.address_of(@_QMmmEarr) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<7xf32>>
    %1 = acc.create varPtr(%0 : !fir.ref<!fir.array<7xf32>>) structured(false) name("arr") -> !fir.ref<!fir.array<7xf32>>
    %2 = acc.declare_enter dataOperands(%1 : !fir.ref<!fir.array<7xf32>>)
    acc.terminator
  }
  acc.global_ctor @_QMmmEotherEarr_acc_ctor {
    %0 = fir.address_of(@_QMmmEotherEarr) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<3xf32>>
    %1 = acc.create varPtr(%0 : !fir.ref<!fir.array<3xf32>>) structured(false) name("other_arr") -> !fir.ref<!fir.array<3xf32>>
    %2 = acc.declare_enter dataOperands(%1 : !fir.ref<!fir.array<3xf32>>)
    acc.terminator
  }
  acc.global_dtor @_QMmmEarr_acc_dtor {
    %0 = fir.address_of(@_QMmmEarr) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<7xf32>>
    %1 = acc.getdeviceptr varPtr(%0 : !fir.ref<!fir.array<7xf32>>) dataClause(acc_create) structured(false) name("arr") -> !fir.ref<!fir.array<7xf32>>
    acc.declare_exit dataOperands(%1 : !fir.ref<!fir.array<7xf32>>)
    acc.delete accPtr(%1 : !fir.ref<!fir.array<7xf32>>) dataClause(acc_create) structured(false) name("arr")
    acc.terminator
  }
  acc.global_dtor @_QMmmEotherEarr_acc_dtor {
    %0 = fir.address_of(@_QMmmEotherEarr) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<3xf32>>
    %1 = acc.getdeviceptr varPtr(%0 : !fir.ref<!fir.array<3xf32>>) dataClause(acc_create) structured(false) name("other_arr") -> !fir.ref<!fir.array<3xf32>>
    acc.declare_exit dataOperands(%1 : !fir.ref<!fir.array<3xf32>>)
    acc.delete accPtr(%1 : !fir.ref<!fir.array<3xf32>>) dataClause(acc_create) structured(false) name("other_arr")
    acc.terminator
  }
  func.func private @_FortranAProgramEndStatement()
  llvm.func @_FortranACUFRegisterAllocator() attributes {sym_visibility = "private"}
  llvm.func internal @__cudaFortranConstructor() {
    llvm.call @_FortranACUFRegisterAllocator() : () -> ()
    %0 = cuf.register_module @cuda_device_mod -> !llvm.ptr
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor], priorities = [0 : i32], data = [#llvm.zero]
}
