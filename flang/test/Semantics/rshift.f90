! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Check that a call to RSHIFT is transformed to SHIFTA.

subroutine test_default_integer()
  integer :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,j)
  k = rshift(16, 2)
!CHECK: k=4_4
end

subroutine test_integer1()
  integer(1) :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,int(j,kind=4))
  print *, rshift(8_1, 2)
!CHECK: PRINT *, 2_1
end

subroutine test_integer2()
  integer(2) :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,int(j,kind=4))
  print *, rshift(8_2, 2)
!CHECK: PRINT *, 2_2
end

subroutine test_integer4()
  integer(4) :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,j)
  print *, rshift(8_4, 2)
!CHECK: PRINT *, 2_4
end

subroutine test_integer8()
  integer(8) :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,int(j,kind=4))
  print *, rshift(-16_8, 2)
!CHECK: PRINT *, -4_8
end

subroutine test_integer16()
  integer(16) :: i, j, k
  k = rshift(i, j)
!CHECK: k=shifta(i,int(j,kind=4))
  print *, rshift(8_16, 2)
!CHECK: PRINT *, 2_16
end
