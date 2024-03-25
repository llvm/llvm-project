! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
subroutine s1
  integer x
  block
    import, none
    !ERROR: 'x' from host scoping unit is not accessible due to IMPORT
    x = 1
  end block
end

subroutine s2
  block
    import, none
    !ERROR: 'y' from host scoping unit is not accessible due to IMPORT
    y = 1
  end block
end

subroutine s3
  implicit none
  integer :: i, j
  block
    import, none
    !ERROR: No explicit type declared for 'i'
    real :: a(16) = [(i, i=1, 16)]
    real :: b(16)
    !ERROR: No explicit type declared for 'j'
    data(b(j), j=1, 16) / 16 * 0.0 /
  end block
end

subroutine s4
  real :: j
  !ERROR: Must have INTEGER type, but is REAL(4)
  real :: a(16) = [(x, x=1, 16)]
  real :: b(16)
  !ERROR: Must have INTEGER type, but is REAL(4)
  data(b(j), j=1, 16) / 16 * 0.0 /
end

subroutine s5
  implicit none
  data x/1./
  !PORTABILITY: 'x' appeared in a DATA statement before its type was declared under IMPLICIT NONE(TYPE)
  real x
end
