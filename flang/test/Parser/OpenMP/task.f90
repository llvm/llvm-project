! REQUIRES: openmp_runtime
! RUN: %flang_fc1 %openmp_flags -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50  %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse -fopenmp -fopenmp-version=50  %s | FileCheck --ignore-case --check-prefix="CHECK-UNPARSE" %s

!CHECK: OmpBlockDirective -> llvm::omp::Directive = task
!CHECK: OmpClauseList -> OmpClause -> Detach -> OmpDetachClause -> OmpObject -> Designator -> DataRef -> Name = 'event'

!CHECK-UNPARSE: INTEGER(KIND=8_4) event
!CHECK-UNPARSE: !$OMP TASK  DETACH(event)
!CHECK-UNPARSE: !$OMP END TASK
subroutine task_detach
  use omp_lib
  implicit none
  integer(kind=omp_event_handle_kind) :: event
  !$omp task detach(event)
  !$omp end task
end subroutine
