! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=CHECK-HLFIR

program test
  integer a(2)
  integer b(2)
  integer c(2)
  integer z(10)
  integer :: k = 10
  integer :: j

!! When a scalar array element is used, the array element is replaced with a temprorary so it is correctly lowered as an Integer
!$omp do reduction (+: a(2))
  do i = 1,2
    a(2) = a(2) + i
  end do
!$omp end do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %17#0 -> %arg1 : !fir.ref<i32>) {
! CHECK-HLFIR: %53:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %54:2 = hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_a(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %53#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %55 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %56 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %57 = arith.addi %55, %56 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %57 to %54#0 : i32, !fir.ref<i32>

!! Ensure that consective reduction clauses can be correctly processed in the same block
!$omp do reduction (+: b(2))
  do i = 1,3
    b(2) = b(2) + i
  end do
!$omp end do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %19#0 -> %arg1 : !fir.ref<i32>) {
! CHECK-HLFIR: %53:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %54:2 = hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_b(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %53#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %55 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %56 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %57 = arith.addi %55, %56 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %57 to %54#0 : i32, !fir.ref<i32>

!! Ensure that we can reuse the same array element later on. This will use the same symbol as the previous use of a(2) for the temporary value
!$omp do reduction (+: a(2))
  do i = 1,4
    a(2) = a(2) + i
!! We need to make sure that for the array element that has not been reduced, this does not get replaced with a temp
    a(1) = a(2)
  end do
!$omp end do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %17#0 -> %arg1 : !fir.ref<i32>) {
! CHECK-HLFIR: %53:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %54:2 = hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_a(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %53#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %55 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %56 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %57 = arith.addi %55, %56 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %57 to %54#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %58 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %c1 = arith.constant 1 : index
! CHECK-HLFIR-NEXT: %59 = hlfir.designate %3#0 (%c1)  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK-HLFIR-NEXT: hlfir.assign %58 to %59 : i32, !fir.ref<i32>

!! Check that multiple reductions work correctly
!$omp parallel do reduction (+:a(2), b(2))
  do i=1,10
    a(2) = a(2) + i
    b(2) = b(2) + i
  end do
!$omp end parallel do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %17#0 -> %arg1, @add_reduction_i32 %19#0 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK-HLFIR: %53:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %54:2 = hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_a(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %55:2 = hlfir.declare %arg2 {uniq_name = "_QFEreduction_temp_b(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg3 to %53#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %56 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %57 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %58 = arith.addi %56, %57 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %58 to %54#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %59 = fir.load %55#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %60 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %61 = arith.addi %59, %60 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %61 to %55#0 : i32, !fir.ref<i32>

!! Check that when the identifier for the elment comes from a variable, this get replaced
!$omp parallel do reduction (+: c(j))
  do i=1,10
    c(j) = c(j) + i
  end do
!$omp end parallel do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %21#0 -> %arg1 : !fir.ref<i32>) {
! CHECK-HLFIR: %53:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: %54:2 = hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_c(j)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %53#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: %55 = fir.load %54#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %56 = fir.load %53#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: %57 = arith.addi %55, %56 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %57 to %54#0 : i32, !fir.ref<i32>

!! Array Sections will not get changed
  !$omp parallel do reduction(+:z(1:10:1))
  do i=1,10
  end do
  !$omp end parallel do
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(byref @add_reduction_byref_box_10xi32 %54 -> %arg1 : !fir.ref<!fir.box<!fir.array<10xi32>>>) {

end program test