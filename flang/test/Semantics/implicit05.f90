! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
subroutine s
  !ERROR: 'a' does not follow 'b' alphabetically
  implicit integer(b-a)
end
