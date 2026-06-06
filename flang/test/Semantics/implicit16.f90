! RUN: %python %S/test_errors.py %s %flang_fc1
interface
!ERROR: No explicit type declared for 'a'
  subroutine s(a)
    implicit none
  end
!ERROR: No explicit type declared for 'f'
  function f()
    implicit none
  end
end interface
end
