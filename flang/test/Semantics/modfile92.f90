! Verify that a module re-exporting a module with a declare reduction by a plain
! USE still hides the names that it renamed to an ambiguous name.

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/origin.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/other.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/facade.f90
! RUN: FileCheck --input-file=%t/mf92_facade.mod %s
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/host.f90

! CHECK-DAG: {{^}}use mf92_origin{{$}}
! CHECK-DAG: {{^}}use mf92_origin,only:z=>x{{$}}
! CHECK-DAG: {{^}}use mf92_other,only:z=>y{{$}}

!--- origin.f90
module mf92_origin
  type :: dt
    integer :: val = 0
  end type
  !$omp declare reduction(+:dt:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=dt(0))
  integer, parameter :: x = 1
end module

!--- other.f90
module mf92_other
  integer, parameter :: y = 2
end module

!--- facade.f90
module mf92_facade
  use mf92_origin, z => x
  use mf92_other, z => y
end module

!--- host.f90
program host
  implicit none
  integer, parameter :: x = 99
  call show
contains
  subroutine show
    use mf92_facade
    ! x must still be the host's x (99): the facade renamed the module's x.
    integer, parameter :: chk(x - 98) = [5]
  end subroutine
end program
