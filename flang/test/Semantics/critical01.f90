! RUN: %python %S/test_errors.py %s %flang_fc1
!C1117

subroutine test1(a, i)
  integer i
  real a(10)
  one: critical
    if (a(i) < 0.0) then
      a(i) = 20.20
    end if
  !ERROR: CRITICAL construct name mismatch
  end critical two
end subroutine test1

subroutine test2(a, i)
  integer i
  real a(10)
  critical
    if (a(i) < 0.0) then
      a(i) = 20.20
    end if
  !ERROR: CRITICAL construct name unexpected
  end critical two
end subroutine test2
