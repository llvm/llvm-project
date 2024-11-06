! RUN: bbc -fcuda -emit-hlfir %s -o - | fir-opt --cuf-add-constructor | FileCheck %s

program main
  real, device :: ahost(10)
end

! CHECK: llvm.func @_FortranACUFRegisterAllocator() attributes {sym_visibility = "private"}
! CHECK-LABEL: llvm.func internal @__cudaFortranConstructor() {
! CHECK:   llvm.call @_FortranACUFRegisterAllocator() : () -> ()
! CHECK:   llvm.return
! CHECK: }
! CHECK: llvm.mlir.global_ctors {ctors = [@__cudaFortranConstructor], priorities = [0 : i32]}
