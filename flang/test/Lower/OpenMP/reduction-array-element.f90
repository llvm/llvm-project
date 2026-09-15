! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s --implicit-check-not=omp.reduction.element

! Array-element reductions currently use the whole-array boxed reduction path.
! Keep lowering coverage for the supported constructs so that this limitation
! does not hide failures to compile them.

! CHECK: omp.declare_reduction @[[BOX_RED:add_reduction_byref_box_4xi32]] byref_element_type({{.*}}) : !fir.ref<!fir.box<!fir.array<4xi32>>>

subroutine reduction_literal(a, n)
  integer :: a(4), n
!$omp parallel do reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_literal
! CHECK: omp.wsloop {{.*}} reduction(byref @[[BOX_RED]] {{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFreduction_literalEa"} : (!fir.ref<!fir.box<!fir.array<4xi32>>>)
! CHECK: fir.load %{{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>
! CHECK: hlfir.designate %{{.*}} (%c2) {{.*}} -> !fir.ref<i32>

subroutine reduction_multiple(a, n)
  integer :: a(4), n
!$omp parallel do reduction(+: a(2), a(3))
  do i = 1, n
    a(2) = a(2) + i
    a(3) = a(3) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_multiple
! CHECK: omp.wsloop {{.*}} reduction(byref @[[BOX_RED]] {{.*}}, byref @[[BOX_RED]] {{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>, !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: hlfir.designate %{{.*}} (%c2) {{.*}} -> !fir.ref<i32>
! CHECK: hlfir.designate %{{.*}} (%c3) {{.*}} -> !fir.ref<i32>

subroutine reduction_arrays(a, b, n)
  integer :: a(4), b(4), n
!$omp parallel do reduction(+: a(2), b(2))
  do i = 1, n
    a(2) = a(2) + b(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_arrays
! CHECK: omp.wsloop {{.*}} reduction(byref @[[BOX_RED]] {{.*}}, byref @[[BOX_RED]] {{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>, !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFreduction_arraysEa"} : (!fir.ref<!fir.box<!fir.array<4xi32>>>)
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFreduction_arraysEb"} : (!fir.ref<!fir.box<!fir.array<4xi32>>>)
! CHECK: hlfir.designate %{{.*}} (%c2{{.*}}) {{.*}} -> !fir.ref<i32>

subroutine reduction_variable(a, n, j)
  integer :: a(4), n, j
!$omp parallel do reduction(+: a(j))
  do i = 1, n
    a(j) = a(j) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_variable
! CHECK: omp.wsloop {{.*}} reduction(byref @[[BOX_RED]] {{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFreduction_variableEa"} : (!fir.ref<!fir.box<!fir.array<4xi32>>>)
! CHECK: hlfir.designate %{{.*}} (%{{.*}}) {{.*}} -> !fir.ref<i32>

subroutine reduction_do_simd(a, n)
  integer :: a(4), n
!$omp parallel do simd reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_do_simd
! CHECK: omp.wsloop reduction(byref @[[BOX_RED]] {{.*}} -> %[[WSARG:.*]] : !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: omp.simd {{.*}} reduction(byref @[[BOX_RED]] %[[WSARG]] -> %{{.*}} : !fir.ref<!fir.box<!fir.array<4xi32>>>) {
! CHECK: hlfir.designate %{{.*}} (%c2) {{.*}} -> !fir.ref<i32>
