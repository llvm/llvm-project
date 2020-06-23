! RUN: %S/test_errors.sh %s %t %f18
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
  real :: i, j
  !ERROR: Must have INTEGER type, but is REAL(4)
  real :: a(16) = [(i, i=1, 16)]
  real :: b(16)
  !ERROR: Must have INTEGER type, but is REAL(4)
  data(b(j), j=1, 16) / 16 * 0.0 /
end
