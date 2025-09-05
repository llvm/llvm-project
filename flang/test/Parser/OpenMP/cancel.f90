!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck %s

! CHECK: SUBROUTINE parallel
subroutine parallel
! CHECK: !$OMP PARALLEL
  !$omp parallel
! CHECK: !$OMP CANCEL PARALLEL
    !$omp cancel parallel
! CHECK: !$OMP END PARALLEL
  !$omp end parallel
! CHECK: END SUBROUTINE
end subroutine

! CHECK: SUBROUTINE sections
subroutine sections
! CHECK: !$OMP PARALLEL SECTIONS
  !$omp parallel sections
! CHECK: !$OMP CANCEL SECTIONS
    !$omp cancel sections
! CHECK: !$OMP END PARALLEL SECTIONS
  !$omp end parallel sections
! CHECK: END SUBROUTINE
end subroutine

! CHECK: SUBROUTINE loop
subroutine loop
! CHECK: !$OMP PARALLEL DO
  !$omp parallel do
! CHECK: DO i=1_4,10_4
    do i=1,10
! CHECK: !$OMP CANCEL DO
      !$omp cancel do
! CHECK: END DO
    enddo
! CHECK: !$OMP END PARALLEL DO
  !$omp end parallel do
! CHECK: END SUBROUTINE
end subroutine

! CHECK: SUBROUTINE taskgroup
subroutine taskgroup
! CHECK: !$OMP TASKGROUP
  !$omp taskgroup
! CHECK: !$OMP TASK
    !$omp task
! CHECK: !$OMP CANCEL TASKGROUP
      !$omp cancel taskgroup
! CHECK: !$OMP END TASK
    !$omp end task
! CHECK: !$OMP END TASKGROUP
  !$omp end taskgroup
! CHECK: END SUBROUTINE
end subroutine
