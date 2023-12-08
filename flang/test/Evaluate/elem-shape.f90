! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Ensure that optional arguments aren't used to fold SIZE() or SHAPE()
module m
 contains
  subroutine sub(x,y)
    real :: x(:), y(:)
    optional x
    !CHECK: PRINT *, int(size(y,dim=1,kind=8),kind=4)
    print *, size(f(x,y))
  end
  elemental function f(x,y)
    real, intent(in) :: x, y
    optional x
    f = y
  end
end
