! RUN: %python %S/test_errors.py %s %flang_fc1
! Test coarray association in CHANGE TEAM statement

subroutine s1
  use iso_fortran_env
  type(team_type) :: t
  complex :: x[*]
  real :: y[*]
  real :: z
  ! OK
  change team(t, x[*] => y)
  end team
  ! C1116
  !ERROR: Selector in coarray association must name a coarray
  change team(t, x[*] => 1)
  end team
  !ERROR: Selector in coarray association must name a coarray
  change team(t, x[*] => z)
  end team
end

subroutine s2
  use iso_fortran_env
  type(team_type) :: t
  real :: y[10,*], y2[*], x[*]
  ! C1113
  !ERROR: The codimensions of 'x' have already been declared
  change team(t, x[10,*] => y, x[*] => y2)
  end team
end
