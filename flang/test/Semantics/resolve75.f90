! RUN: %S/test_errors.sh %s %t %f18
! C726 The length specified for a character statement function or for a 
! statement function dummy argument of type character shall be a constant 
! expression.
subroutine s()
  implicit character(len=3) (c)
  implicit character(len=*) (d)
  stmtFunc1 (x) = x * 32
  cStmtFunc2 (x) = "abc"
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
  cStmtFunc3 (dummy) = "abc"
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
  dStmtFunc3 (x) = "abc"
end subroutine s
