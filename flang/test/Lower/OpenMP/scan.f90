! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK: omp.wsloop private({{.*}}) reduction(mod: inscan, @add_reduction_i32 %{{.*}} -> %[[RED_ARG_1:.*]] : {{.*}}) {
! CHECK: %[[RED_DECL_1:.*]]:2 = hlfir.declare %[[RED_ARG_1]]
! CHECK: omp.scan inclusive(%[[RED_DECL_1]]#1 : {{.*}})

subroutine inclusive_scan(a, b, n)
 implicit none
 integer a(:), b(:)
 integer x, k, n

 !$omp parallel do reduction(inscan, +: x)
 do k = 1, n
   x = x + a(k)
   !$omp scan inclusive(x)
   b(k) = x
 end do
end subroutine inclusive_scan


! CHECK: omp.wsloop private({{.*}}) reduction(mod: inscan, @add_reduction_i32 %{{.*}} -> %[[RED_ARG_2:.*]] : {{.*}}) {
! CHECK: %[[RED_DECL_2:.*]]:2 = hlfir.declare %[[RED_ARG_2]]
! CHECK: omp.scan exclusive(%[[RED_DECL_2]]#1 : {{.*}})
subroutine exclusive_scan(a, b, n)
 implicit none
 integer a(:), b(:)
 integer x, k, n

 !$omp parallel do reduction(inscan, +: x)
 do k = 1, n
   x = x + a(k)
   !$omp scan exclusive(x)
   b(k) = x
 end do
end subroutine exclusive_scan
