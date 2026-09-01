! RUN: bbc -fcuda -emit-fir %s -o - | fir-opt --cuf-add-constructor | FileCheck %s

program main
  real, allocatable, device :: ahost(:)
end

! CHECK-LABEL: llvm.func internal @__cudaFortranConstructor() {
! CHECK:   llvm.call @_FortranACUFRegisterAllocator() : () -> ()
! CHECK:   llvm.return
! CHECK: }
! CHECK: llvm.func @_FortranACUFRegisterAllocator() attributes {sym_visibility = "private"}
! CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor], priorities = [0 : i32], data = [#llvm.zero]
