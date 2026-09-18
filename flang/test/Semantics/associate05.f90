! RUN: %python %S/test_errors.py %s %flang_fc1
! The lower bounds of an associate name whose selector is a reference to a
! function with a POINTER result are those of the pointer's target, which are
! not known at compile time.  LBOUND() of such a name must therefore not be
! folded to a constant.  A non-pointer function result still has default
! lower bounds.
module m
  real, target :: tgt(2:4)
 contains
  function f_ptr()
    real, pointer :: f_ptr(:)
    f_ptr => tgt
  end function
  function f_val()
    real :: f_val(2:4)
    f_val = 0.
  end function
  subroutine test
    associate (fp => f_ptr(), fv => f_val())
      block
        !ERROR: Must be a constant value
        integer, parameter :: k1 = lbound(fp, 1)
        integer, parameter :: k2 = lbound(fv, 1)
      end block
    end associate
  end subroutine
end module
