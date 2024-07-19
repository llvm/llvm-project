! This test checks lowering of OpenMP order clause.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK-LABEL:   func.func @_QPsimd_order() {
subroutine simd_order
   !CHECK: omp.simd order(reproducible:concurrent) {
   !$omp simd order(concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.simd order(reproducible:concurrent) {
   !$omp simd order(reproducible:concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.simd order(unconstrained:concurrent) {
   !$omp simd order(unconstrained:concurrent)
   do i = 1, 10
   end do
end subroutine simd_order

!CHECK-LABEL:   func.func @_QPdo_order() {
subroutine do_order
   !CHECK: omp.wsloop order(reproducible:concurrent) {
   !$omp do order(concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.wsloop order(reproducible:concurrent) {
   !$omp do order(reproducible:concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.wsloop order(unconstrained:concurrent) {
   !$omp do order(unconstrained:concurrent)
   do i = 1, 10
   end do
end subroutine do_order

!CHECK-LABEL:   func.func @_QPdo_simd_order() {
subroutine do_simd_order
   !CHECK: omp.wsloop order(reproducible:concurrent) {
   !$omp do simd order(concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.wsloop order(reproducible:concurrent) {
   !$omp do simd order(reproducible:concurrent)
   do i = 1, 10
   end do
   !CHECK: omp.wsloop order(unconstrained:concurrent) {
   !$omp do simd order(unconstrained:concurrent)
   do i = 1, 10
   end do
end subroutine do_simd_order

!CHECK-LABEL:   func.func @_QPdo_simd_order_parallel() {
subroutine do_simd_order_parallel
   !CHECK: omp.parallel {
   !CHECK: omp.wsloop order(reproducible:concurrent) {
   !$omp parallel do simd order(reproducible:concurrent)
   do i = 1, 10
   end do
end subroutine do_simd_order_parallel


subroutine distribute_order
   !CHECK: omp.distribute order(reproducible:concurrent) {
   !$omp teams distribute order(concurrent)
   do i=1,10
   end do
   !CHECK: omp.distribute order(reproducible:concurrent) {
   !$omp teams distribute order(reproducible:concurrent)
   do i=1,10
   end do
   !CHECK: omp.distribute order(unconstrained:concurrent) {
   !$omp teams distribute order(unconstrained:concurrent)
   do i = 1, 10
   end do
end subroutine
