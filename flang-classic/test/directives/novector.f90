! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLE-METADATA
! RUN: %flang -Hx,59,2 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=IGNORE-DIRECTIVES
! RUN: %flang -Hx,59,2 -S -emit-llvm -O2 %s -o - | FileCheck %s
! RUN: %flang -S -emit-llvm -O2 %s -o - | FileCheck %s -check-prefix=NOVECTOR

subroutine add(arr1,arr2,arr3,N)
  integer :: i,N
  integer :: arr1(N)
  integer :: arr2(N)
  integer :: arr3(N)

  !dir$ novector
  do i = 1, N
    arr3(i) = arr1(i) - arr2(i)
  end do
end subroutine
! DISABLE-METADATA: !"llvm.loop.vectorize.enable", i1 false
! IGNORE-DIRECTIVES-NOT: !"llvm.loop.vectorize.enable", i1 false
! CHECK: load <[[VF:[0-9]+]] x i32>
! CHECK: sub {{.*}} <[[VF]] x i32>
! CHECK: store <[[VF]] x i32>
! NOVECTOR-NOT: <{{[0-9]+}} x {{.+}}>
