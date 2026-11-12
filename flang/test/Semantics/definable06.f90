! RUN: %python %S/test_errors.py %s %flang_fc1
module m
 contains
  elemental subroutine inout(x)
    integer, intent(inout) :: x
  end
  subroutine test
    integer :: x(2)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Variable has a vector subscript with a duplicated element
    x([1,1]) = 0
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: Variable has a vector subscript with a duplicated element
    call inout(x([(mod(j-1,2)+1,j=1,10)]))
    !ERROR: Input variable 'x' is not definable
    !BECAUSE: Variable has a vector subscript with a duplicated element
    read (*,*) x([2,2])
  end
end

