! RUN: %python %S/test_errors.py %s %flang_fc1

module m1
  logical :: l
contains
  subroutine test_relational_character(a, b)
    character(4,kind=1) :: a
    character(4,kind=2) :: b
    !ERROR: CHARACTER operands do not have same KIND
    l = a == b
  end
end
