! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

! C934
! If type-spec appears, it shall specify a type with which each
! allocate-object is type compatible.
! Issue #78939: allocatable object has a non-defined character length.
! This should also be an error when the length is defined by a parameter
! in a module.

module m1
  integer::nn=1
  integer,parameter::np=1
end module m1

program main
  use m1
  character(nn),pointer::cns
  character(np),pointer::c1s
  !ERROR: Character length of allocatable object in ALLOCATE must be the same as the type-spec
  allocate(character(2)::cns)
  !ERROR: Character length of allocatable object in ALLOCATE must be the same as the type-spec
  allocate(character(2)::c1s)
end program main
