! Test runtime behavior of DO SIMD with firstprivate and lastprivate on same variable
! This is the reproducer from issue #168306

! REQUIRES: openmp-runtime

! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
! RUN: %flang -fopenmp %s -o %t && %t | FileCheck %s

! LLVM-LABEL: define {{.*}} @_QQmain
program main
  integer :: a
  integer :: i
  
  a = 10
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 1
     ! Inside loop: a should be 10 (from firstprivate initialization)
     ! CHECK: main1 : a = 10
     print *, "main1 : a = ", a
     a = 20
  end do
  !$omp end do simd
  ! After loop: a should be 20 (from lastprivate copy-out)
  ! CHECK: main2 : a = 20
  print *, "main2 : a = ", a
  
  call sub
  ! CHECK: pass
  print *, 'pass'
end program main

subroutine sub
  integer :: a
  integer :: i
  
  a = 10
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 1
     ! Inside loop: a should be 10 (from firstprivate initialization)
     ! CHECK: sub1  : a = 10
     print *, "sub1  : a = ", a
     a = 20
  end do
  !$omp end do simd
  ! After loop: a should be 20 (from lastprivate copy-out)
  ! CHECK: sub2  : a = 20
  print *, "sub2  : a = ", a
end subroutine sub
