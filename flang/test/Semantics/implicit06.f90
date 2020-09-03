! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  implicit integer(a-c)
  !ERROR: More than one implicit type specified for 'c'
  implicit real(c-g)
end

subroutine s2
  implicit integer(a-c)
  implicit real(8)(d)
  !ERROR: More than one implicit type specified for 'a'
  implicit integer(f), real(a)
end
