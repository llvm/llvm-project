! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!
! Test that when an explicit default DECLARE MAPPER is defined for a type (newdata)
! whose only component is a pointer, and that type is used as an allocatable component
! of another type (newvec), the compiler-generated implicit mapper for newvec correctly
! references the user-defined mapper for newdata rather than synthesising a new
! implicit one that would silently skip the pointer component.
!
! An explicit `map(s)` on a newvec variable is the code-path that triggers implicit
! mapper generation for newvec (unlike an implicit target capture, which suppresses
! mapper generation for plain non-allocatable/non-pointer derived types).

module m
  type :: newdata
    real(8), pointer :: value
  end type
  !$omp declare mapper(default: newdata :: v) map(tofrom: v%value)

  type :: newvec
    integer :: len
    type(newdata), allocatable :: data(:)
  end type
end module

program main
  use m
  implicit none
  type(newvec) :: s

  ! Explicit map(s) triggers implicit mapper generation for newvec.
  !$omp target data map(s)
    s%len = 0
  !$omp end target data
end program

! The user-defined default mapper for newdata must have tofrom map for v%value.
! CHECK-DAG: omp.declare_mapper @_QQMmnewdata_omp_default_mapper
! CHECK-DAG: map_clauses(tofrom)
! CHECK-DAG: {name = "v%value"}

! The implicit default mapper for newvec must reference the user-defined newdata
! mapper by its canonical name.  Before the fix this referenced a separately-
! synthesised bogus mapper (@_QMmTnewdata_omp_default_mapper) that had no maps
! for pointer components, causing a null-address device fault at runtime.
! CHECK-DAG: omp.declare_mapper @_QQMmnewvec_omp_default_mapper
! CHECK-DAG: mapper(@_QQMmnewdata_omp_default_mapper)