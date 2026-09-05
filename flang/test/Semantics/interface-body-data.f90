! RUN: %python %S/test_errors.py %s %flang_fc1

interface
  subroutine s
    integer :: x
    !ERROR: A DATA statement may not appear in an interface body
    data x /1/
  end subroutine
end interface

end