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
