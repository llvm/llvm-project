! RUN: %python %S/test_symbols.py %s %flang_fc1
! Test handling of pernicious case in which it is conformant Fortran
! to use the name of a function in a CALL statement.  Almost all
! other compilers produce bogus errors for this case and/or crash.

!DEF: /m Module
module m
contains
 !DEF: /m/foo PUBLIC (Function) Subprogram
 function foo()
  !DEF: /m/bar PUBLIC (Subroutine) Subprogram
  !DEF: /m/foo/foo EXTERNAL, POINTER (Subroutine) ProcEntity
  procedure(bar), pointer :: foo
  !REF: /m/bar
  !DEF: /m/foo/baz EXTERNAL, POINTER (Subroutine) ProcEntity
  procedure(bar), pointer :: baz
  !REF: /m/foo/foo
  !REF: /m/bar
  foo => bar
  !REF: /m/foo/foo
  call foo
  !DEF: /m/baz PUBLIC (Function) Subprogram
  entry baz()
  !REF: /m/foo/baz
  !REF: /m/bar
  baz => bar
  !REF: /m/foo/baz
  call baz
 end function
 !REF: /m/bar
 subroutine bar
  print *, "in bar"
 end subroutine
end module
!DEF: /DEMO MainProgram
program DEMO
 !REF: /m
 use :: m
 !DEF: /DEMO/bar (Subroutine) Use
 !DEF: /DEMO/p EXTERNAL, POINTER (Subroutine) ProcEntity
 procedure(bar), pointer :: p
 !REF: /DEMO/p
 !DEF: /DEMO/foo (Function) Use
 p => foo()
 !REF: /DEMO/p
 call p
end program
