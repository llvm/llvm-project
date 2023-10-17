! RUN: %python %S/test_errors.py %s %flang_fc1
program main
  type t
    character(4), pointer :: p
  end type
  character(5), target :: buff = "abcde"
  type(t) x
  !ERROR: Target type CHARACTER(KIND=1,LEN=5_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=4_8)
  x = t(buff)
  !ERROR: Target type CHARACTER(KIND=1,LEN=3_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=4_8)
  x = t(buff(3:))
  !ERROR: Target type CHARACTER(KIND=1,LEN=5_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=4_8)
  x%p => buff
  !ERROR: Target type CHARACTER(KIND=1,LEN=3_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=4_8)
  x%p => buff(1:3)
end
