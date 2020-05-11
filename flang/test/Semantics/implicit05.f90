! RUN: %S/test_errors.sh %s %t %f18
subroutine s
  !ERROR: 'a' does not follow 'b' alphabetically
  implicit integer(b-a)
end
