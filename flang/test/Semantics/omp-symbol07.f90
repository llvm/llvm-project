! RUN: %S/test_symbols.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell

! Generic tests
!   1. subroutine or function calls should not be fixed for DSA or DMA

!DEF: /foo (Function) Subprogram REAL(4)
!DEF: /foo/rnum ObjectEntity REAL(4)
function foo(rnum)
  !REF: /foo/rnum
  real rnum
  !REF: /foo/rnum
  rnum = rnum+1.
end function foo
!DEF: /function_call_in_region EXTERNAL (Subroutine) Subprogram
subroutine function_call_in_region
  implicit none
  !DEF: /function_call_in_region/foo (Function) ProcEntity REAL(4)
  real foo
  !DEF: /function_call_in_region/a ObjectEntity REAL(4)
  real :: a = 0.
  !DEF: /function_call_in_region/b ObjectEntity REAL(4)
  real :: b = 5.
  !$omp parallel  default(none) private(a) shared(b)
  !DEF: /function_call_in_region/Block1/a (OmpPrivate) HostAssoc REAL(4)
  !REF: /function_call_in_region/foo
  !REF: /function_call_in_region/b
  a = foo(b)
  !$omp end parallel
  !REF: /function_call_in_region/a
  !REF: /function_call_in_region/b
  print *, a, b
end subroutine function_call_in_region
!DEF: /mm MainProgram
program mm
  !REF: /function_call_in_region
  call function_call_in_region
end program mm
