! One combined reduction(foo : a, b, c) clause over mixed-shape allocatables with
! non-default lower bounds (b(6:9), c(3,8:9)) must lower.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s

subroutine test_udr_combined()
  implicit none
  integer :: i
  integer, allocatable :: a, b(:), c(:,:)

  !$omp declare reduction (foo : integer : omp_out = omp_out + omp_in) &
  !$omp & initializer (omp_priv = 0)

  allocate(a, b(6:9), c(3, 8:9))
  a = 0
  b = 0
  c = 0

  !$omp parallel do reduction(foo : a, b, c)
  do i = 1, 10
    a = a + i
    b = b + i
    c = c + i
  end do
end subroutine

! One boxed op per shape is synthesized from the single scalar base op.
! CHECK-DAG: omp.declare_reduction @{{.*}}foo_byref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-DAG: omp.declare_reduction @{{.*}}foo_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG: omp.declare_reduction @{{.*}}foo_byref_box_heap_UxUxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK-DAG: omp.declare_reduction @{{.*}}foo_i32 : i32

! The one combined clause binds all three boxed ops in a single wsloop.
! CHECK-LABEL: func.func @_QPtest_udr_combined
! CHECK:         omp.wsloop {{.*}}reduction(byref @{{.*}}foo_byref_box_heap_i32 {{.*}}byref @{{.*}}foo_byref_box_heap_Uxi32 {{.*}}byref @{{.*}}foo_byref_box_heap_UxUxi32 {{.*}}!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
