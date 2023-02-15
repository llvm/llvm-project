! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  integer, target :: n
 contains
  function ptr()
    integer, pointer :: ptr
    ptr => n
  end
  subroutine s1(p)
    integer, pointer, intent(in) :: p
  end
  subroutine s2(p)
    integer, pointer, intent(in out) :: p
  end
end

program test
  use m
  integer, pointer :: p
  p => ptr() ! ok
  ptr() = 1 ! ok
  call s1(ptr()) ! ok
  call s1(null()) ! ok
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
  !BECAUSE: 'ptr()' is not a definable pointer
  call s2(ptr())
end
