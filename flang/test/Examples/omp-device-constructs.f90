! Check flang-omp-report --femit-yaml for omp-device-constructs.f90

! REQUIRES: plugins, examples, shell

!RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp %s -o - | FileCheck %s

! Check OpenMP clause validity for the following directives:
!     2.10 Device constructs
program main

  real(8) :: arrayA(256), arrayB(256)
  integer :: N

  arrayA = 1.414
  arrayB = 3.14
  N = 256

  !$omp target map(arrayA)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target device(0)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target defaultmap(tofrom:scalar)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp teams num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !$omp target map(tofrom:a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target data device(0) map(to:a)
  do i = 1, N
    a = 3.14
  enddo
  !$omp end target data

end program main

! CHECK: ---
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            18
! CHECK-NEXT:   construct:       target
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          map
! CHECK-NEXT:       details:         arraya
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            24
! CHECK-NEXT:   construct:       target
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          device
! CHECK-NEXT:       details:         '0'
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            30
! CHECK-NEXT:   construct:       target
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          defaultmap
! CHECK-NEXT:       details:         'tofrom:scalar'
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            36
! CHECK-NEXT:   construct:       teams
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          default
! CHECK-NEXT:       details:         shared
! CHECK-NEXT:     - clause:          num_teams
! CHECK-NEXT:       details:         '3'
! CHECK-NEXT:     - clause:          private
! CHECK-NEXT:       details:         i
! CHECK-NEXT:     - clause:          shared
! CHECK-NEXT:       details:         a
! CHECK-NEXT:     - clause:          thread_limit
! CHECK-NEXT:       details:         '10'
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            42
! CHECK-NEXT:   construct:       target
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          map
! CHECK-NEXT:       details:         'tofrom:a'
! CHECK-NEXT: - file:            '{{[^"]*}}omp-device-constructs.f90'
! CHECK-NEXT:   line:            48
! CHECK-NEXT:   construct:       target data
! CHECK-NEXT:   clauses:
! CHECK-NEXT:     - clause:          device
! CHECK-NEXT:       details:         '0'
! CHECK-NEXT:     - clause:          map
! CHECK-NEXT:       details:         'to:a'
! CHECK-NEXT: ...
