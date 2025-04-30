! RUN: %flang -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-WRONG-CLAUSE

subroutine add(arr1,arr2,arr3,N)
  integer :: i,N
  integer :: arr1(N)
  integer :: arr2(N)
  integer :: arr3(N)

  !dir$ vector never
  do i = 1, N
    arr3(i) = arr1(i) - arr2(i)
  end do
end subroutine
! CHECK-WRONG-CLAUSE: F90-W-0603-Unsupported clause specified for the vector directive. Only the always and vectorlength clauses are supported.
