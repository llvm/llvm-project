!RUN: %python %S/test_errors.py %s %flang_fc1

function good() result(pptr)
  procedure(), pointer :: pptr
  external whatever
  pptr => whatever
end

function bad1() result(res1)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() res1
end

!ERROR: Procedure 'res2' is referenced before being sufficiently defined in a context where it must be so
function bad2() result(res2)
  !ERROR: EXTERNAL attribute not allowed on 'res2'
  external res2
end

function s(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(S(dd))
end

! A procedure POINTER result is valid, but a reference to such a function
! is still not acceptable as an intrinsic array argument.
function s1(x) result(i)
  procedure(), pointer :: i
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(S1(dd))
end

function s2(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Output item must not be a procedure
  print *, s2(dd)
end

function s3(x) result(i)
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: i
  !ERROR: Selector may not be a procedure
  select type (y => s3(dd))
  end select
end
