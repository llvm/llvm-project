! This test checks lowering of OpenMP linear clause

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPtestdolinear() {
subroutine testDoLinear()
   implicit none
   integer :: i
   integer :: A(10)
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFtestdolinearEa"}
!CHECK: %[[S2:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]](%[[S2]]) {uniq_name = "_QFtestdolinearEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK: %[[C2:.*]] = arith.constant 2 : i32
!CHECK: omp.wsloop linear(%[[A_DECL]]#1 = %[[C2]] : !fir.ref<!fir.array<10xi32>>) {
!$omp do linear(A:2)
   do i = 1, 10
      A(i) = i
   end do
!$omp end do
end subroutine testDoLinear

! CHECK-LABEL: func.func @_QPtestsimdlinear() {
subroutine testSimdLinear()
   implicit none
   integer :: i
   integer :: A(10)
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFtestsimdlinearEa"}
!CHECK: %[[S2:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]](%[[S2]]) {uniq_name = "_QFtestsimdlinearEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK: %[[C2:.*]] = arith.constant 2 : i32
!CHECK:   omp.simd linear(%[[A_DECL]]#1 = %[[C2]] : !fir.ref<!fir.array<10xi32>>) {
!$omp simd linear(A:2)
   do i = 1, 10
      A(i) = i
   end do
!$omp end simd
end subroutine testSimdLinear

! CHECK-LABEL: func.func @_QPtestdosimdlinear() {
subroutine testDoSimdLinear()
   implicit none
   integer :: i
   integer :: A(10)
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFtestdosimdlinearEa"}
!CHECK: %[[S2:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]](%[[S2]]) {uniq_name = "_QFtestdosimdlinearEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK: %[[C2:.*]] = arith.constant 2 : i32
!CHECK: omp.wsloop {
!CHECK:   omp.simd linear(%[[A_DECL]]#1 = %[[C2]] : !fir.ref<!fir.array<10xi32>>) {
!$omp do simd linear(A:2)
   do i = 1, 10
      A(i) = i
   end do
!$omp end do simd
end subroutine testDoSimdLinear
