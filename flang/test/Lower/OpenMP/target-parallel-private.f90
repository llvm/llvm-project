! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

!===============================================================================
! `private` clause on `target parallel`
!===============================================================================

subroutine target_parallel_private()
integer, dimension(3) :: i
!$omp target parallel private(i)
!$omp end target parallel
end subroutine

! CHECK: omp.private {type = private} @[[PRIVATIZER:.*]] : {{.*}}

! CHECK: omp.target {{.*}} {
! CHECK:   omp.parallel private(@[[PRIVATIZER]] %{{.*}} -> %{{.*}} : {{.*}}) {
! CHECK:   }
! CHECK: }
