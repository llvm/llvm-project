!RUN: %flang_fc1 -emit-llvm -fopenmp %openmp_flags -fopenmp-version=52 %s -o - | FileCheck %s

!Check that this doesn't crash.

!CHECK-LABEL: define void @f00_()
!CHECK: call i1 @_FortranAioOutputDerivedType

subroutine f00
  use iso_c_binding
  type(c_ptr) :: x

!$omp target data use_device_ptr(x) if(.false.)
  print *, x
!$omp end target data
end

!CHECK-LABEL: define void @f01_()
!CHECK: call i1 @_FortranAioOutputInteger32
subroutine f01
  integer :: x

!$omp target data use_device_addr(x) if(.false.)
  print *, x
!$omp end target data
end
