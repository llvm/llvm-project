! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s
! Test that we do not make a private copy of the critical name

!CHECK: Global scope:
!CHECK-NEXT: MN: MainProgram
!CHECK-NEXT: k2 (OmpCriticalLock): Unknown

!CHECK:  MainProgram scope: MN
!CHECK-NEXT:    j size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK-NEXT:    OtherConstruct scope:
!CHECK-NEXT:      j (OmpPrivate): HostAssoc
!CHECK-NOT:  k2

program mn
  integer :: j
  j=2
  !$omp parallel default(private)
    !$omp critical(k2)
    j=200
    !$omp end critical(k2)
  !$omp end parallel
end 
