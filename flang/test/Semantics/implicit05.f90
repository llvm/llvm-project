! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
subroutine s
  !ERROR: 'a' does not follow 'b' alphabetically
  implicit integer(b-a)
end
