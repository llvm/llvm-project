! RUN: %python %S/test_errors.py %s %flang_fc1
! DATA on components of a parameterized derived type instance with length type
! parameters is not yet supported.
program test
  type ut(n)
    integer, len :: n
    character(n) :: sar(2)
    integer :: i
  end type
  type outer
    type(ut(1)) :: in
  end type
  type(ut(1)) pdt
  type(outer) o
  !ERROR: not yet implemented: DATA statement initialization of a component in a parameterized derived type instance
  data pdt%sar/'o','k'/
  !ERROR: not yet implemented: DATA statement initialization of a component in a parameterized derived type instance
  data pdt%i/7/
  !ERROR: not yet implemented: DATA statement initialization of a component in a parameterized derived type instance
  data o%in%i/7/
end program

! A kind-only PDT has no length parameters and must still be accepted.
subroutine test_kind
  type ut(k)
    integer, kind :: k
    integer(k) :: i
    character(2) :: c(2)
  end type
  type(ut(4)) pdt
  data pdt%i/7/, pdt%c/'a','b'/
end subroutine
