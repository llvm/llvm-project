! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s
  !ERROR: 'a' does not follow 'b' alphabetically
  implicit integer(b-a)
end
