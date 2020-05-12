! RUN: %S/test_errors.sh %s %t %f18
!C1119

subroutine test1(a, i)
  integer i
  real a(10)
  critical
    if (a(i) < 0.0) then
      a(i) = 20.20
      !ERROR: Control flow escapes from CRITICAL
      goto 20
    end if
  end critical
20 a(i) = -a(i)
end subroutine test1

subroutine test2(i)
  integer i
  critical
    !ERROR: Control flow escapes from CRITICAL
    if (i) 10, 10, 20
    10 i = i + 1
  end critical
20 i = i - 1
end subroutine test2

subroutine test3(i)
  integer i
  critical
    !ERROR: Control flow escapes from CRITICAL
    goto (10, 10, 20) i
    10 i = i + 1
  end critical
20 i = i - 1
end subroutine test3
