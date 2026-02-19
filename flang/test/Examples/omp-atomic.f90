! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport%pluginext -plugin flang-omp-report -fopenmp %s -o - | FileCheck %s

! Check OpenMP 2.13.6 atomic Construct

  a = 1.0
  !$omp parallel num_threads(4) shared(a)
  !$omp atomic seq_cst, read
  b = a

  !$omp atomic seq_cst write
  a = b
  !$omp end atomic

  !$omp atomic capture seq_cst
  b = a
  a = a + 1
  !$omp end atomic

  !$omp atomic
  a = a + 1
  !$omp end parallel
end

! CHECK:---
! CHECK-NEXT:- file:            '{{[^"]*}}omp-atomic.f90'
! CHECK-NEXT:  line:            [[@LINE-19]]
! CHECK-NEXT:  construct:       atomic
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      read
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:    - clause:      seq_cst
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-atomic.f90'
! CHECK-NEXT:  line:            [[@LINE-24]]
! CHECK-NEXT:  construct:       atomic
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      seq_cst
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:    - clause:      write
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-atomic.f90'
! CHECK-NEXT:  line:            [[@LINE-28]]
! CHECK-NEXT:  construct:       atomic
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      capture
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:    - clause:      seq_cst
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-atomic.f90'
! CHECK-NEXT:  line:            [[@LINE-31]]
! CHECK-NEXT:  construct:       atomic
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-atomic.f90'
! CHECK-NEXT:  line:            [[@LINE-48]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      num_threads
! CHECK-NEXT:      details:     '4'
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     a
! CHECK-NEXT:...
