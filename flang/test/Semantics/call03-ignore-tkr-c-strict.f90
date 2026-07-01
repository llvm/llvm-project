! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Assumed-type dummies must enforce F2023 15.5.2.5 p2 for derived types with
! PDT, TBP, or FINAL unless bind(C), ignore_tkr(c), and descriptor passing
! all apply.

module m
  type :: tbp
   contains
    procedure :: binding => subr
  end type
  type :: pdt(n)
    integer, len :: n
  end type
  type :: final_typ
   contains
    final :: cleanup
  end type

 contains

  subroutine subr(this)
    class(tbp), intent(in) :: this
  end subroutine
  subroutine cleanup(this)
    type(final_typ), intent(inout) :: this
  end subroutine
  subroutine cfi(x) bind(c)
    type(*), dimension(..) :: x
  end subroutine
  subroutine not_cfi(x)
    type(*), dimension(..) :: x
!dir$ ignore_tkr(c) x
  end subroutine
  subroutine not_descriptor(x) bind(c)
    type(*) :: x(*)
!dir$ ignore_tkr(c) x
  end subroutine
end module

program main
  use m
  type(tbp) :: x
  type(tbp), dimension(1) :: arr
  type(pdt(1)) :: y
  type(final_typ) :: z
  !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have type-bound procedure 'binding'
  call cfi(x)
  !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have a parameterized derived type
  call cfi(y)
  !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have derived type 'final_typ' with FINAL subroutine 'cleanup'
  call cfi(z)
  !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have type-bound procedure 'binding'
  call not_cfi(x)
  !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have type-bound procedure 'binding'
  call not_descriptor(arr)
end program
