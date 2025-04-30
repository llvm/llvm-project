! RUN: %flang -O0 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-WRONG-CLAUSE --implicit-check-not "llvm.loop.vectorize"
subroutine func1(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ simd simdlen(garbage)
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func1
! CHECK-WRONG-CLAUSE: 90-W-0804-Unsupported argument in the simdlen clause; allowed values are positive numeric literal.

subroutine func2(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ simd simdlen(3,
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func2
! CHECK-WRONG-CLAUSE: 90-W-0804-Unsupported argument in the simdlen clause; allowed values are positive numeric literal.

subroutine func3(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ simd simdlen(3 5 7
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func3
! CHECK-WRONG-CLAUSE: 90-W-0804-Unsupported argument in the simdlen clause; allowed values are positive numeric literal.

subroutine func4(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ simd simdlenX
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func4
! CHECK-WRONG-CLAUSE: F90-W-0605-Unsupported clause specified for the dir simd directive. Only the simdlen clause is supported, other directive clauses will be ignored.
