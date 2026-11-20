!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Dummy arguments are marked noalias. Check that at -O2 the load and the store
! of a dummy argument still happen inside the critical section and are not
! moved across the OpenMP runtime calls (issue #201759).

!RUN: %flang_fc1 -emit-llvm -fopenmp -O2 %s -o - | FileCheck %s

subroutine sub(tmp)
  real(8) :: tmp
  integer :: i
  !$omp single
  do i = 1, 10
    !$omp critical
    tmp = tmp + real(i, 8)
    !$omp end critical
  end do
  !$omp end single
end subroutine

!CHECK-LABEL: define void @sub_(
!CHECK-SAME: ptr noalias {{.*}}%[[TMP:[0-9]+]])
!CHECK: call void @__kmpc_critical(
!CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr {{.*}}%[[TMP]])
!CHECK-NEXT: load double, ptr %[[TMP]]
!CHECK-NOT: __kmpc_end_critical
!CHECK: store double %{{.*}}, ptr %[[TMP]]
!CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr {{.*}}%[[TMP]])
!CHECK-NEXT: call void @__kmpc_end_critical(
