! RUN: %flang_fc1 -fsyntax-only %s
! Test that ignore_tkr(c) on an assumed-type bind(C) descriptor dummy
! relaxes F2023 15.5.2.5 p2 restrictions for opaque CFI argument passing.

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
!dir$ ignore_tkr(c) x
  end subroutine
end module

program main
  use m
  type(tbp) :: x
  type(pdt(1)) :: y
  type(final_typ) :: z
  call cfi(x)
  call cfi(y)
  call cfi(z)
end program
