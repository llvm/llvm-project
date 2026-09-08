!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | tco -test-gen | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | tco -test-gen | FileCheck %s

! Test that a bare '!$omp declare target' inside an interface body that
! appears in a *named* main program does not incorrectly mark the main
! program (_QQmain) as a declare-target function while still correctly
! marking the declared subroutine (sub_a) as device_type(any).

! CHECK-NOT: llvm.func @_QQmain{{.*}}device_type = any
! CHECK-NOT: llvm.func @_QQmain{{.*}}device_type = nohost
! CHECK: llvm.func @_QPsub_a{{.*}}device_type = any, {{.*}}sym_visibility = "private"

program named_main
  interface
    subroutine sub_a(x)
      implicit none
      !$omp declare target
      integer, intent(inout) :: x
    end subroutine sub_a
  end interface
  integer :: v = 0
  !$omp target
    call sub_a(v)
  !$omp end target
end program named_main
