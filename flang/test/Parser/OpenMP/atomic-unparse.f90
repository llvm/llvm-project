! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck %s

program main
   implicit none
   integer :: i, j = 10
!READ
!$omp atomic read
   i = j
!$omp atomic seq_cst read
   i = j
!$omp atomic read seq_cst
   i = j
!$omp atomic release read
   i = j
!$omp atomic read release
   i = j
!$omp atomic acq_rel read
   i = j
!$omp atomic read acq_rel
   i = j
!$omp atomic acquire read
   i = j
!$omp atomic read acquire
   i = j
!$omp atomic relaxed read
   i = j
!$omp atomic read relaxed
   i = j

! WRITE
!$omp atomic write
   i = j
!$omp atomic seq_cst write
   i = j
!$omp atomic write seq_cst
   i = j
!$omp atomic release write
   i = j
!$omp atomic write release
   i = j
!$omp atomic acq_rel write
   i = j
!$omp atomic write acq_rel
   i = j
!$omp atomic acquire write
   i = j
!$omp atomic write acquire
   i = j
!$omp atomic relaxed write
   i = j
!$omp atomic write relaxed
   i = j

!UPDATE
!$omp atomic update
   i = j
!$omp atomic seq_cst update
   i = j
!$omp atomic update seq_cst
   i = j
!$omp atomic release update
   i = j
!$omp atomic update release
   i = j
!$omp atomic acq_rel update
   i = j
!$omp atomic update acq_rel
   i = j
!$omp atomic acquire update
   i = j
!$omp atomic update acquire
   i = j
!$omp atomic relaxed update
   i = j
!$omp atomic update relaxed
   i = j

!CAPTURE
!$omp atomic capture
   i = j
   i = j
!$omp end atomic
!$omp atomic seq_cst capture
   i = j
   i = j
!$omp end atomic
!$omp atomic capture seq_cst
   i = j
   i = j
!$omp end atomic
!$omp atomic release capture
   i = j
   i = j
!$omp end atomic
!$omp atomic capture release
   i = j
   i = j
!$omp end atomic
!$omp atomic acq_rel capture
   i = j
   i = j
!$omp end atomic
!$omp atomic capture acq_rel
   i = j
   i = j
!$omp end atomic
!$omp atomic acquire capture
   i = j
   i = j
!$omp end atomic
!$omp atomic capture acquire
   i = j
   i = j
!$omp end atomic
!$omp atomic relaxed capture
   i = j
   i = j
!$omp end atomic
!$omp atomic capture relaxed
   i = j
   i = j
!$omp end atomic

!ATOMIC
!$omp atomic
   i = j
!$omp atomic seq_cst
   i = j
!$omp atomic release
   i = j
!$omp atomic acq_rel
   i = j
!$omp atomic acquire
   i = j
!$omp atomic relaxed
   i = j

end program main
!CHECK-LABEL: PROGRAM main

!READ

!CHECK: !$OMP ATOMIC READ
!CHECK: !$OMP ATOMIC SEQ_CST READ
!CHECK: !$OMP ATOMIC READ SEQ_CST
!CHECK: !$OMP ATOMIC RELEASE READ
!CHECK: !$OMP ATOMIC READ RELEASE
!CHECK: !$OMP ATOMIC ACQ_REL READ
!CHECK: !$OMP ATOMIC READ ACQ_REL
!CHECK: !$OMP ATOMIC ACQUIRE READ
!CHECK: !$OMP ATOMIC READ ACQUIRE
!CHECK: !$OMP ATOMIC RELAXED READ
!CHECK: !$OMP ATOMIC READ RELAXED

!WRITE

!CHECK: !$OMP ATOMIC WRITE
!CHECK: !$OMP ATOMIC SEQ_CST WRITE
!CHECK: !$OMP ATOMIC WRITE SEQ_CST
!CHECK: !$OMP ATOMIC RELEASE WRITE
!CHECK: !$OMP ATOMIC WRITE RELEASE
!CHECK: !$OMP ATOMIC ACQ_REL WRITE
!CHECK: !$OMP ATOMIC WRITE ACQ_REL
!CHECK: !$OMP ATOMIC ACQUIRE WRITE
!CHECK: !$OMP ATOMIC WRITE ACQUIRE
!CHECK: !$OMP ATOMIC RELAXED WRITE
!CHECK: !$OMP ATOMIC WRITE RELAXED

!UPDATE

!CHECK: !$OMP ATOMIC UPDATE
!CHECK: !$OMP ATOMIC SEQ_CST UPDATE
!CHECK: !$OMP ATOMIC UPDATE SEQ_CST
!CHECK: !$OMP ATOMIC RELEASE UPDATE
!CHECK: !$OMP ATOMIC UPDATE RELEASE
!CHECK: !$OMP ATOMIC ACQ_REL UPDATE
!CHECK: !$OMP ATOMIC UPDATE ACQ_REL
!CHECK: !$OMP ATOMIC ACQUIRE UPDATE
!CHECK: !$OMP ATOMIC UPDATE ACQUIRE
!CHECK: !$OMP ATOMIC RELAXED UPDATE
!CHECK: !$OMP ATOMIC UPDATE RELAXED

!CAPTURE

!CHECK: !$OMP ATOMIC CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC SEQ_CST CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC CAPTURE SEQ_CST
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC RELEASE CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC CAPTURE RELEASE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC ACQ_REL CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC CAPTURE ACQ_REL
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC ACQUIRE CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC CAPTURE ACQUIRE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC RELAXED CAPTURE
!CHECK: !$OMP END ATOMIC
!CHECK: !$OMP ATOMIC CAPTURE RELAXED
!CHECK: !$OMP END ATOMIC

!ATOMIC
!CHECK: !$OMP ATOMIC
!CHECK: !$OMP ATOMIC SEQ_CST
!CHECK: !$OMP ATOMIC RELEASE
!CHECK: !$OMP ATOMIC ACQ_REL
!CHECK: !$OMP ATOMIC ACQUIRE
!CHECK: !$OMP ATOMIC RELAXED
