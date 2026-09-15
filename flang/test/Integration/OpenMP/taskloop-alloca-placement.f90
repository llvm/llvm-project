!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! This test is to ensure that allocas generated inside the loop body of a
! taskloop are located correctly inside the omp.taskloop.context but not inside
! the loop itself.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | tco -test-gen | FileCheck %s

! CHECK-LABEL: llvm.func @_QPtest_taskloop(
subroutine test_taskloop(lb,ub,step,a)
  integer :: lb, ub, step
  integer, allocatable :: a(:)

  !$omp taskloop shared(a)
! CHECK:         omp.taskloop.context private({{.*}}) {
! CHECK:           llvm.alloca
! CHECK:           omp.taskloop.wrapper {
  do i = lb,ub,step
! CHECK-NOT:         llvm.alloca
    a(i) = i
  enddo
  !$omp end taskloop
! CHECK:             omp.yield
! CHECK-NEXT:      }
end subroutine
