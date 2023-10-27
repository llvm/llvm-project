! RUN: %python %S/test_errors.py %s %flang_fc1
! PDT sensitivity of FINAL subroutines
module m
  type :: pdt(k)
    integer, kind :: k
   contains
    final :: finalArr, finalElem
  end type
 contains
  subroutine finalArr(x)
    type(pdt(1)), intent(in out) :: x(:)
  end
  elemental subroutine finalElem(x)
    type(pdt(3)), intent(in out) :: x
  end
end

program test
  use m
  type(pdt(1)) x1(1)
  type(pdt(2)) x2(1)
  type(pdt(3)) x3(1)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Variable 'x1([INTEGER(8)::1_8])' has a vector subscript and cannot be finalized by non-elemental subroutine 'finalarr'
  x1([1]) = pdt(1)()
  x2([1]) = pdt(2)() ! ok, doesn't match either
  x3([1]) = pdt(3)() ! ok, calls finalElem
end
