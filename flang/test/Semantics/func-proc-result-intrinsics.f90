! RUN: %python %S/test_errors.py %s %flang_fc1
! A reference to a function whose result is a plain procedure (invalid:
! a function result is either a variable or a procedure pointer, F2023
! 19.3.3), passed to various intrinsics: each must produce an accurate
! argument error instead of an assertion failure.

function s1(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'a=' may not be a procedure
  print *, rank(s1(dd))
end

function s2(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'source=' may not be a procedure
  print *, shape(s2(dd))
end

function s3(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(s3(dd), 1)
end

function s4(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'a=' may not be a procedure
  print *, storage_size(s4(dd))
end

function s5(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'source=' may not be a procedure
  print *, transfer(s5(dd), 0)
end

function s6(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, lbound(s6(dd), 1)
end
