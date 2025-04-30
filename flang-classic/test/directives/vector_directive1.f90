! RUN: %flang -c %s 2>&1 -o - | FileCheck %s --check-prefix=CHECK-NO-CLAUSE
! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s --check-prefix=METADATA
! RUN: %flang -S -emit-llvm -O2 %s 2>&1 -o - | FileCheck %s

subroutine add(arr1,arr2,arr3,N)
  integer :: i,N
  integer :: arr1(N)
  integer :: arr2(N)
  integer :: arr3(N)

  !dir$ vector
  do i = 1, N
    arr3(i) = arr1(i) - arr2(i)
  end do
end subroutine
! CHECK-NO-CLAUSE-NOT: F90-S-0602
! CHECK-NO-CLAUSE-NOT: F90-S-0603

! METADATA: !"llvm.loop.vectorize.enable", i1 true
! CHECK: load <[[VF:[0-9]+]] x i32>
! CHECK: store <[[VF]] x i32>

