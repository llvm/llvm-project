!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Tests the OMPIRBuilder can handle multiple privatization regions that contain
! multiple BBs (for example, for allocatables).

! RUN: %flang -S -emit-llvm -fopenmp -mmlir --enable-delayed-privatization \
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
