! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) team
loop1: do j = 1, 1
    goto 1 ! ok
1 construct2: change team (team)
      goto 2 ! ok
      exit construct2 ! ok
      !ERROR: EXIT must not leave a CHANGE TEAM statement
      exit loop1
      !ERROR: EXIT must not leave a CHANGE TEAM statement
      exit
      !ERROR: CYCLE must not leave a CHANGE TEAM statement
      cycle
      !ERROR: RETURN statement is not allowed in a CHANGE TEAM construct
      return
      !ERROR: Control flow escapes from CHANGE TEAM
      goto 3
      !ERROR: Control flow escapes from CHANGE TEAM
      write(*,*,err=3)
2   end team construct2
3 end do loop1
end
