!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Verify that host OpenMPOpt infers noalias for Flang's aggregate capture
! container, allowing both captured addresses to be loaded before an opaque
! call.
!
! RUN: %flang_fc1 -O2 -fopenmp -emit-llvm %s -o - | FileCheck %s

subroutine capture_lifetime(first, second)
  integer(8), intent(in) :: first, second

  interface
    subroutine opaque()
    end subroutine
    subroutine use_value(value)
      integer(8), value :: value
    end subroutine
  end interface

  !$omp parallel shared(first, second)
  call opaque()
  call use_value(first)
  call use_value(second)
  !$omp end parallel
end subroutine

! CHECK-LABEL: define internal void @capture_lifetime_..omp_par(
! CHECK-SAME: ptr noalias readonly captures(none) [[CAPTURES:%.*]])
! CHECK: [[FIRST:%.*]] = load ptr, ptr [[CAPTURES]], align 8
! CHECK-NEXT: [[SECOND_FIELD:%.*]] = getelementptr i8, ptr [[CAPTURES]], i64 8
! CHECK-NEXT: [[SECOND:%.*]] = load ptr, ptr [[SECOND_FIELD]], align 8
! CHECK-NEXT: tail call void @opaque_()
! CHECK: load i64, ptr [[FIRST]], align 8
! CHECK: load i64, ptr [[SECOND]], align 8
