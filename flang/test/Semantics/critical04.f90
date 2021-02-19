! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: Control flow escapes from CRITICAL

subroutine test1(a, i)
  integer i
  real a(10)
  critical
    if (a(i) < 0.0) then
      a(i) = 20.20
      goto 20
    end if
20 a(i) = -a(i)
  end critical
end subroutine test1

subroutine test2(i)
  integer i
  critical
    if (i) 10, 10, 20
10  i = i + 1
20  i = i - 1
  end critical
end subroutine test2

subroutine test3(i)
  integer i
  critical
    goto (10, 10, 20) i
10  i = i + 1
20  i = i - 1
  end critical
end subroutine test3
