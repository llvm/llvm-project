! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  real mobj
 contains
  subroutine msubr
  end subroutine
end module
program main
  use m
  !PORTABILITY: Name 'main' declared in a main program should not have the same name as the main program
  pointer main
  !ERROR: Cannot change POINTER attribute on use-associated 'mobj'
  pointer mobj
  !ERROR: Cannot change POINTER attribute on use-associated 'msubr'
  pointer msubr
  !ERROR: 'inner' cannot have the POINTER attribute
  pointer inner
  real obj
  !ERROR: 'ip' may not have both the POINTER and PARAMETER attributes
  integer, parameter :: ip = 123
  pointer ip
  type dt; end type
  !ERROR: 'dt' cannot have the POINTER attribute
  pointer dt
  interface generic
    subroutine extsub
    end subroutine
  end interface
  !ERROR: 'generic' cannot have the POINTER attribute
  pointer generic
  namelist /nml/ obj
  !ERROR: 'nml' cannot have the POINTER attribute
  pointer nml
 contains
  subroutine inner
  end subroutine
end
