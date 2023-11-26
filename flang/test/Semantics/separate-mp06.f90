! RUN: %python %S/test_errors.py %s %flang_fc1
! Structural equivalence of derived type definitions
module m
  interface
    module subroutine s1(x)
      type :: nonseq
        integer :: n
      end type
      type(nonseq), intent(in) :: x
    end subroutine
    module subroutine s2(x)
      type :: seq
        sequence
        integer :: n
      end type
      type(seq), intent(in) :: x
    end subroutine
    module subroutine s3(x)
      type :: chlen
        sequence
        character(2) :: s
      end type
      type(chlen), intent(in) :: x
    end subroutine
    module subroutine s4(x)
      !ERROR: A sequence type may not have type parameters
      type :: pdt(k)
        integer, kind :: k
        sequence
        real(k) :: a
      end type
      type(pdt(4)), intent(in) :: x
    end subroutine
  end interface
end module

submodule(m) sm
 contains
  module subroutine s1(x)
    type :: nonseq
      integer :: n
    end type
    !ERROR: Dummy argument 'x' has type nonseq; the corresponding argument in the interface body has distinct type nonseq
    type(nonseq), intent(in) :: x
  end subroutine
  module subroutine s2(x) ! ok
    type :: seq
      sequence
      integer :: n
    end type
    type(seq), intent(in) :: x
  end subroutine
  module subroutine s3(x)
    type :: chlen
      sequence
      character(3) :: s ! note: length is 3, not 2
    end type
    !ERROR: Dummy argument 'x' has type chlen; the corresponding argument in the interface body has distinct type chlen
    type(chlen), intent(in) :: x
  end subroutine
  module subroutine s4(x)
    !ERROR: A sequence type may not have type parameters
    type :: pdt(k)
      integer, kind :: k
      sequence
      real(k) :: a
    end type
    !ERROR: Dummy argument 'x' has type pdt(k=4_4); the corresponding argument in the interface body has distinct type pdt(k=4_4)
    type(pdt(4)), intent(in) :: x
  end subroutine
end submodule

program main
  use m
  type :: nonseq
    integer :: n
  end type
  type :: seq
    sequence
    integer :: n
  end type
  type :: chlen
    sequence
    character(2) :: s
  end type
  !ERROR: A sequence type may not have type parameters
  type :: pdt(k)
    integer, kind :: k
    sequence
    real(k) :: a
  end type
  !ERROR: Actual argument type 'nonseq' is not compatible with dummy argument type 'nonseq'
  call s1(nonseq(1))
  call s2(seq(1)) ! ok
  call s3(chlen('ab')) ! ok, matches interface
  !ERROR: Actual argument type 'pdt(k=4_4)' is not compatible with dummy argument type 'pdt(k=4_4)'
  call s4(pdt(4)(3.14159))
end program
