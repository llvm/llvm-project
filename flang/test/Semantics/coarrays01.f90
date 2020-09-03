! RUN: %S/test_errors.sh %s %t %f18
! Test selector and team-value in CHANGE TEAM statement

! OK
subroutine s1
  use iso_fortran_env, only: team_type
  type(team_type) :: t
  real :: y[10,*]
  change team(t, x[10,*] => y)
  end team
  form team(1, t)
end

subroutine s2
  use iso_fortran_env
  type(team_type) :: t
  real :: y[10,*], y2[*], x[*]
  ! C1113
  !ERROR: Selector 'y' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, x2[*] => y)
  end team
  !ERROR: Selector 'x' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, x2[*] => x)
  end team
  !ERROR: Coarray 'y' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, y[*] => y2)
  end team
end

subroutine s3
  type :: team_type
  end type
  type :: foo
    real :: a
  end type
  type(team_type) :: t1
  type(foo) :: t2
  type(team_type) :: t3(3)
  real :: y[10,*]
  ! C1114
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  change team(t1, x[10,*] => y)
  end team
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  change team(t2, x[10,*] => y)
  end team
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  change team(t2%a, x[10,*] => y)
  end team
  !ERROR: Must be a scalar value, but is a rank-1 array
  change team(t3, x[10,*] => y)
  end team
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(1, t1)
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(2, t2)
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(2, t2%a)
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(3, t3(2))
  !ERROR: Must be a scalar value, but is a rank-1 array
  form team(3, t3)
end

subroutine s4
  use iso_fortran_env, only: team_type
  complex :: z
  integer :: i, j(10)
  type(team_type) :: t, t2(2)
  form team(i, t)
  !ERROR: Must be a scalar value, but is a rank-1 array
  form team(1, t2)
  !ERROR: Must have INTEGER type, but is COMPLEX(4)
  form team(z, t)
  !ERROR: Must be a scalar value, but is a rank-1 array
  form team(j, t)
end
