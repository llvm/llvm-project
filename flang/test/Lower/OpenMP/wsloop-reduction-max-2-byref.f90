! RUN: bbc -emit-hlfir -fopenmp --force-byref-reduction -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --force-byref-reduction -o - %s 2>&1 | FileCheck %s

! CHECK: omp.wsloop reduction(byref @max_byref_i32
! CHECK: arith.cmpi sgt
! CHECK: arith.select

module m1
  intrinsic max
end module m1
program main
  use m1, ren=>max
  n=0
  !$omp parallel do reduction(ren:n)
  do i=1,100
     n=max(n,i)
  end do
  if (n/=100) print *,101
  print *,'pass'
end program main
