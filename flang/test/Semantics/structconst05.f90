! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that PDT instance structure constructors can be folded to constants
module m1
  type :: pdt(k)
    integer, len :: k
    character(len=k) :: x, y = "def"
  end type
  type(pdt(4)) :: v = pdt(4)("abc")
end module
