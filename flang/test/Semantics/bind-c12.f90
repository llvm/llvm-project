! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: A dummy procedure to an interoperable procedure must also be interoperable
subroutine subr(e) bind(c)
  external e
end
