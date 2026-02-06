! RUN: %python %S/test_errors.py %s %flang_fc1
pure subroutine puresub
  intrinsic sleep, chdir, get_command
  character(80) str
  !ERROR: Procedure 'impureexternal' referenced in pure subprogram 'puresub' must be pure too
  call impureExternal ! implicit interface
  !ERROR: Procedure 'sleep' referenced in pure subprogram 'puresub' must be pure too
  call sleep(1) ! intrinsic subroutine, debatably impure
  !ERROR: Procedure 'chdir' referenced in pure subprogram 'puresub' must be pure too
  call chdir('.') ! "dual" function/subroutine, impure
end
