! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s --implicit-check-not=omp.reduction.element

! Regression test for reductions on array *sections* (e.g. a(2:96)).
!
! An array section has rank > 0, so it must be lowered using the boxed
! whole-array reduction (@add_reduction_byref_box_*) and the section applied
! via hlfir.designate inside the region. It must NOT be routed through the
! single-element path (uniq_name = "omp.reduction.element"), which only supports
! rank-0 references and otherwise aborts in PrivateReductionUtils with
! "creating reduction/privatization init region for unsupported type".

subroutine reduction_array_section(a, n)
  integer :: a(100), n
!$omp parallel do reduction(+: a(2:96))
  do i = 1, n
    a(2:96) = a(2:96) + i
  end do
end subroutine

! CHECK: omp.declare_reduction @[[RED:add_reduction_byref_box_100xi32]] : !fir.ref<!fir.box<!fir.array<100xi32>>>

! CHECK-LABEL: func.func @_QPreduction_array_section
! CHECK: omp.wsloop {{.*}} reduction(byref @[[RED]] %{{[0-9]+}} -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.array<100xi32>>>) {
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFreduction_array_sectionEa"}
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.array<100xi32>>>
! CHECK: hlfir.designate %[[BOX]] (%c2:%c96:%c1) {{.*}} -> !fir.ref<!fir.array<95xi32>>

subroutine reduction_array_section_simd(a, n)
  integer :: a(100), n
!$omp parallel do simd reduction(+: a(2:96))
  do i = 1, n
    a(2:96) = a(2:96) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_array_section_simd
! CHECK: omp.wsloop reduction(byref @[[RED]] %{{[0-9]+}} -> %[[WSARG:.*]] : !fir.ref<!fir.box<!fir.array<100xi32>>>) {
! CHECK: omp.simd {{.*}} reduction(byref @[[RED]] %[[WSARG]] -> %[[SIMDARG:.*]] : !fir.ref<!fir.box<!fir.array<100xi32>>>) {
! CHECK: hlfir.declare %[[SIMDARG]] {uniq_name = "_QFreduction_array_section_simdEa"}
