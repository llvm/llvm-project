!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Regression test for an intermittent crash in taskloop LLVM IR codegen.
! A dead saveIP/restoreIP pair in OpenMPIRBuilder::createTaskloop introduced
! stale IRBuilder/debug-location state on the restore path before an
! immediately following SetInsertPoint override. Removing the dead restore
! avoids that restore path entirely.

! RUN: %flang_fc1 -emit-llvm -fopenmp -o - %s | FileCheck %s

! Verify that the taskloop runtime call is present in the generated LLVM IR
! and that the i64 bounds casts required by the ABI are emitted.

! CHECK-LABEL: define void @test_taskloop_(

! CHECK: sext i32 {{.*}} to i64
! CHECK: call void @__kmpc_taskloop(

subroutine test_taskloop(n)
  integer, intent(in) :: n
  integer :: i

  !$omp taskloop
  do i = 1, n
  end do
  !$omp end taskloop
end subroutine test_taskloop
