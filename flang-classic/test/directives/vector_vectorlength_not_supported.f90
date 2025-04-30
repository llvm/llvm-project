! RUN: %flang -O0 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-WRONG-CLAUSE --implicit-check-not "llvm.loop.vectorize"
subroutine func1(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ vector vectorlength(garbage)
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func1
! CHECK-WRONG-CLAUSE: 90-W-0803-Unsupported argument in the vectorlength clause; allowed values are one of: 'fixed', 'scalable', or a numeric literal

subroutine func2(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ vector vectorlength(3,
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func2
! CHECK-WRONG-CLAUSE: 90-W-0803-Unsupported argument in the vectorlength clause; allowed values are one of: 'fixed', 'scalable', or a numeric literal

subroutine func3(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ vector vectorlength(3 5 7
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func3
! CHECK-WRONG-CLAUSE: 90-W-0803-Unsupported argument in the vectorlength clause; allowed values are one of: 'fixed', 'scalable', or a numeric literal

subroutine func4(a, b, m)
  integer :: i, m, a(m), b(m)
  !dir$ vector vectorlengthX
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine func4
! CHECK-WRONG-CLAUSE: F90-W-0603-Unsupported clause specified for the vector directive. Only the always and vectorlength clauses are supported.
