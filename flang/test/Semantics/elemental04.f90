! RUN: %python %S/test_errors.py %s %flang_fc1
! ELEMENTAL dummy argument constraints (C15100) in scopes nested inside
! elemental and non-elemental procedures

elemental subroutine s1(x)
  real, intent(in) :: x
  interface
    subroutine ext1(a)
      real :: a(:) ! ok: ext1 is not elemental
    end
  end interface
  block
    real :: loc(2) ! ok: not a dummy argument
  end block
end

subroutine s2(y)
  real :: y(:) ! ok: s2 is not elemental
  interface
    elemental subroutine ext2(b)
      !ERROR: A dummy argument of an ELEMENTAL procedure must be scalar
      real, intent(in) :: b(:)
    end
  end interface
end
