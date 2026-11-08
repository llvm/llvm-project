!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

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
