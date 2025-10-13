! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
      program main
! CHECK:       k01=1+
! CHECK: !$   &1
      k01=1+
!$   &1

! CHECK: !$    k02=2
! CHECK:       3
! CHECK: !$   &4
!$    k02=2
     +3
!$   +4

! CHECK: !$omp parallel private(k01)
!$omp parallel
!$omp+ private(k01)
!$omp end parallel

! CHECK-NOT: comment
!$omp parallel
!$acc+comment
!$omp end parallel
      end
