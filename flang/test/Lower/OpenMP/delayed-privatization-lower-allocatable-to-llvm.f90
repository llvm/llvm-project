! Tests the OMPIRBuilder can handle multiple privatization regions that contain
! multiple BBs (for example, for allocatables).

! RUN: %flang -S -emit-llvm -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s

subroutine foo(x)
  integer, allocatable :: x, y
!$omp parallel private(x, y)
  x = y
!$omp end parallel
end

! CHECK-LABEL: define void @foo_
! CHECK:         ret void
! CHECK-NEXT:  }

! CHECK-LABEL: define internal void @foo_..omp_par
! CHECK-DAG:     call ptr @malloc
! CHECK-DAG:     call ptr @malloc
! CHECK-DAG:     call void @free
! CHECK-DAG:     call void @free
! CHECK:       }
