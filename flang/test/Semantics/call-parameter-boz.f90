! RUN: %python %S/test_errors.py %s %flang_fc1 -Whollerith-or-character-as-boz
! The extension that passes a CHARACTER actual to a scalar INTEGER dummy as
! if it were BOZ must keep working when the actual is an element of a named
! constant (whose designator is retained for storage association): the
! conversion inspects the folded value.
module m
contains
  subroutine takes_integer(x)
    integer, intent(in) :: x
  end subroutine
end module

subroutine named_constant_element()
  use m
  character(4), parameter :: a(1) = ['abcd']
  !PORTABILITY: passing Hollerith or character literal as if it were BOZ [-Whollerith-or-character-as-boz]
  call takes_integer(a(1))
end subroutine

subroutine literal_and_parenthesized()
  use m
  character(4), parameter :: a(1) = ['abcd']
  !PORTABILITY: passing Hollerith or character literal as if it were BOZ [-Whollerith-or-character-as-boz]
  call takes_integer('abcd')
  !PORTABILITY: passing Hollerith or character literal as if it were BOZ [-Whollerith-or-character-as-boz]
  call takes_integer((a(1)))
end subroutine

subroutine variable_still_rejected()
  use m
  character(4) :: c
  !ERROR: Actual argument type 'CHARACTER(KIND=1,LEN=4_8)' is not compatible with dummy argument type 'INTEGER(4)'
  call takes_integer(c)
end subroutine
