! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Check that a call to LSHIFT is transformed to SHIFTL.

subroutine test_default_integer()
  integer :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,j)
  k = lshift(16, 2)
!CHECK: k=64_4
end

subroutine test_integer1()
  integer(1) :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,int(j,kind=4))
  print *, lshift(8_1, 2)
!CHECK: PRINT *, 32_1
end

subroutine test_integer2()
  integer(2) :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,int(j,kind=4))
  print *, lshift(8_2, 2)
!CHECK: PRINT *, 32_2
end

subroutine test_integer4()
  integer(4) :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,j)
  print *, lshift(8_4, 2)
!CHECK: PRINT *, 32_4
end

subroutine test_integer8()
  integer(8) :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,int(j,kind=4))
  print *, lshift(-16_8, 2)
!CHECK: PRINT *, -64_8
end

subroutine test_integer16()
  integer(16) :: i, j, k
  k = lshift(i, j)
!CHECK: k=shiftl(i,int(j,kind=4))
  print *, lshift(8_16, 2)
!CHECK: PRINT *, 32_16
end
