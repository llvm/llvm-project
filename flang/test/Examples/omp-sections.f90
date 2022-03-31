! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp %s -o - | FileCheck %s

subroutine omp_sections()
  integer :: x
  !$omp sections private(x)
    !$omp section
    call f1()
    !$omp section
    call f2()
  !$omp end sections nowait
end subroutine omp_sections

!CHECK: - file:            {{.*}}
!CHECK:   line:            9
!CHECK:   construct:       section
!CHECK:   clauses:         []
!CHECK: - file:            {{.*}}
!CHECK:   line:            11
!CHECK:   construct:       section
!CHECK:   clauses:         []
!CHECK: - file:            {{.*}}
!CHECK:   line:            7
!CHECK:   construct:       sections
!CHECK:   clauses:
!CHECK:     - clause:          nowait
!CHECK:       details:         ''
!CHECK:     - clause:          private
!CHECK:       details:         x
