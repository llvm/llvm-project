!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine inclusive_scan
 implicit none
 integer, parameter :: n = 100
 integer a(n), b(n)
 integer x, k

 !CHECK: omp.wsloop reduction(mod: inscan, {{.*}}) {
 !$omp parallel do reduction(inscan, +: x)
 do k = 1, n
   x = x + a(k)
   !CHECK: omp.scan inclusive({{.*}})
   !$omp scan inclusive(x)
   b(k) = x
 end do
end subroutine inclusive_scan


subroutine exclusive_scan
 implicit none
 integer, parameter :: n = 100
 integer a(n), b(n)
 integer x, k

 !CHECK: omp.wsloop reduction(mod: inscan, {{.*}}) {
 !$omp parallel do reduction(inscan, +: x)
 do k = 1, n
   x = x + a(k)
   !CHECK: omp.scan exclusive({{.*}})
   !$omp scan exclusive(x)
   b(k) = x
 end do
end subroutine exclusive_scan
