! RUN: bbc -fcuda -emit-fir %s -o - | fir-opt --cuf-add-constructor | FileCheck %s

module m0
  integer, allocatable :: a(:)
end module

! CHECK-LABEL: llvm.func internal @__cudaFortranConstructor() {
! CHECK-NOT: llvm.call @_FortranACUFRegisterAllocator()
