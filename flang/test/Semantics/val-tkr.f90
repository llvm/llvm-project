! RUN: %python %S/test_errors.py %s %flang_fc1
implicit none
interface
  subroutine s(b)
    !dir$ ignore_tkr(tr) b
    real, value :: b
  end
  subroutine s1(b)
    !dir$ ignore_tkr(r) b
    integer, value :: b
  end
end interface
integer :: a(5), a1
! forbid array to scalar with VALUE and ignore_tkr(r)
!ERROR: Array actual argument may not be associated with IGNORE_TKR(R) scalar dummy argument 'b=' with VALUE attribute
call s(a)
!ERROR: Array actual argument may not be associated with IGNORE_TKR(R) scalar dummy argument 'b=' with VALUE attribute
call s1(a)
! allow scalar to scalar with VALUE
call s(a1)
call s1(a(1))
end
