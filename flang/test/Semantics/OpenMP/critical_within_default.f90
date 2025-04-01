! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s
! Test that we do not make a private copy of the critical name

!CHECK:  MainProgram scope: mn
!CHECK-NEXT:    j size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK-NEXT:    OtherConstruct scope:
!CHECK-NEXT:      j (OmpPrivate): HostAssoc
!CHECK-NEXT:      k2 (OmpCriticalLock): Unknown
program mn
  integer :: j
  j=2
  !$omp parallel default(private)
    !$omp critical(k2)
    j=200
    !$omp end critical(k2)
  !$omp end parallel
end 
