! RUN: %python %S/test_symbols.py %s %flang_fc1
! Exercise generic redefinitions in inner procedures with conflicting subprograms.
!DEF: /m Module
module m
 !DEF: /m/generic PUBLIC (Subroutine) Generic
 interface generic
  !DEF: /m/specific1 PUBLIC (Subroutine) Subprogram
  module procedure :: specific1
 end interface
contains
 !REF: /m/specific1
 subroutine specific1
  print *, 1
 end subroutine
 !DEF: /m/specific2 PUBLIC (Subroutine) Subprogram
 subroutine specific2
  print *, 2
 end subroutine
 !DEF: /m/test PUBLIC (Subroutine) Subprogram
 subroutine test
  !REF: /m/specific1
  call generic
 end subroutine
 !DEF: /m/outer PUBLIC (Subroutine) Subprogram
 subroutine outer
  !DEF: /m/outer/inner1 (Subroutine) Subprogram
  call inner1
 contains
  !REF: /m/outer/inner1
  subroutine inner1
   !DEF: /m/outer/inner1/generic (Subroutine) Generic
   interface generic
    !REF: /m/specific2
    module procedure :: specific2
   end interface
   !REF: /m/specific2
   call generic
  end subroutine inner1
 end subroutine outer
end module m
!DEF: /main MainProgram
program main
 !REF: /m
 use :: m
 !REF: /m/specific1
 call generic
 !DEF: /main/inner2 (Subroutine) Subprogram
 call inner2
contains
 !REF: /main/inner2
 subroutine inner2
  !DEF: /main/inner2/generic (Subroutine) Generic
  interface generic
   !DEF: /main/specific2 (Subroutine) Use
   module procedure :: specific2
  end interface
  !REF: /main/specific2
  call generic
 end subroutine inner2
end program
