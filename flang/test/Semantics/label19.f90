! RUN: %python %S/test_errors.py %s %flang_fc1
program main
  use, intrinsic:: iso_fortran_env, only: team_type
  type(team_type) team
  logical :: p = false
1 change team(team)
2 if (p) goto 1 ! ok
  if (p) goto 2 ! ok
  if (p) goto 3 ! ok
  if (p) goto 4 ! ok
  if (p) goto 5 ! ok
3 end team
4 continue
  if (p) goto 1 ! ok
  !ERROR: Label '2' is in a construct that prevents its use as a branch target here
  if (p) goto 2
  !ERROR: Label '3' is in a construct that prevents its use as a branch target here
  if (p) goto 3
5 end
