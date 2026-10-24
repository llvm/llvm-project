! RUN: %python %S/test_errors.py %s %flang_fc1 -fcoarray

! F2023 15.8 p2: A SIMPLE procedure is also PURE
simple subroutine s()
  !ERROR: An image control statement may not appear in a pure subprogram
  sync all
end subroutine