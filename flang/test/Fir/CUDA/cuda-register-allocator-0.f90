! RUN: bbc -fcuda -emit-fir %s -o - | fir-opt --cuf-add-constructor | FileCheck %s

module m0
  integer, device, allocatable :: a(:)
end module

! CHECK-LABEL: llvm.func internal @__cudaFortranConstructor() {
! CHECK: llvm.call @_FortranACUFRegisterAllocator()
