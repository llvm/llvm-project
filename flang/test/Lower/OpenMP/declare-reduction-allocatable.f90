! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s

subroutine test_udr_allocatable()
  implicit none
  integer :: i
  integer, allocatable :: a, b(:), c(:,:)

  !$omp declare reduction (foo : integer : omp_out = omp_out + omp_in) &
  !$omp & initializer (omp_priv = 0)

  allocate(a, b(4), c(3,2))
  a = 0
  b = 0
  c = 0

  !$omp parallel do reduction(foo : a)
  do i = 1, 10
    a = a + i
  end do

  !$omp parallel do reduction(foo : b)
  do i = 1, 10
    b = b + i
  end do

  !$omp parallel do reduction(foo : c)
  do i = 1, 10
    c = c + i
  end do
end subroutine

! CHECK-LABEL: omp.declare_reduction @foo_byref_box_heap_UxUxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK-SAME:  attributes {byref_element_type = !fir.array<?x?xi32>}
! CHECK:       alloc {
! CHECK:         fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
! CHECK:         omp.yield
! CHECK:       } init {
! CHECK:         %[[C0_2D:.*]] = arith.constant 0 : i32
! CHECK:         omp.yield
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[ARG0_2D:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, %[[ARG1_2D:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>):
! CHECK:         %[[LHS_BOX_2D:.*]] = fir.load %[[ARG0_2D]]
! CHECK:         %[[RHS_BOX_2D:.*]] = fir.load %[[ARG1_2D]]
! CHECK:         fir.shape_shift
! CHECK:         fir.do_loop {{.*}} unordered {
! CHECK:           fir.do_loop {{.*}} unordered {
! CHECK:             fir.array_coor %[[LHS_BOX_2D]]
! CHECK:             fir.array_coor %[[RHS_BOX_2D]]
! CHECK:             arith.addi
! CHECK:             fir.store
! CHECK:           }
! CHECK:         }
! CHECK:         omp.yield(%[[ARG0_2D]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
! CHECK:       } cleanup {
! CHECK:         fir.freemem
! CHECK:         omp.yield

! CHECK-LABEL: omp.declare_reduction @foo_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-SAME:  attributes {byref_element_type = !fir.array<?xi32>}
! CHECK:       alloc {
! CHECK:         fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:         omp.yield
! CHECK:       } init {
! CHECK:         %[[C0_1D:.*]] = arith.constant 0 : i32
! CHECK:         omp.yield
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[ARG0_1D:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[ARG1_1D:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:         %[[LHS_BOX_1D:.*]] = fir.load %[[ARG0_1D]]
! CHECK:         %[[RHS_BOX_1D:.*]] = fir.load %[[ARG1_1D]]
! CHECK:         fir.shape_shift
! CHECK:         fir.do_loop {{.*}} unordered {
! CHECK:           fir.array_coor %[[LHS_BOX_1D]]
! CHECK:           fir.array_coor %[[RHS_BOX_1D]]
! CHECK:           arith.addi
! CHECK:           fir.store
! CHECK:         }
! CHECK:         omp.yield(%[[ARG0_1D]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:       } cleanup {
! CHECK:         fir.freemem
! CHECK:         omp.yield

! CHECK-LABEL: omp.declare_reduction @foo_byref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-SAME:  attributes {byref_element_type = i32}
! CHECK:       alloc {
! CHECK:         fir.alloca !fir.box<!fir.heap<i32>>
! CHECK:         omp.yield
! CHECK:       } init {
! CHECK:         %[[C0_S:.*]] = arith.constant 0 : i32
! CHECK:         omp.yield
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[ARG0_S:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[ARG1_S:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:         %[[LHS_BOX_S:.*]] = fir.load %[[ARG0_S]]
! CHECK:         %[[RHS_BOX_S:.*]] = fir.load %[[ARG1_S]]
! CHECK:         %[[LHS_ADDR_S:.*]] = fir.box_addr %[[LHS_BOX_S]]
! CHECK:         %[[RHS_ADDR_S:.*]] = fir.box_addr %[[RHS_BOX_S]]
! CHECK:         fir.load %[[LHS_ADDR_S]]
! CHECK:         fir.load %[[RHS_ADDR_S]]
! CHECK:         arith.addi
! CHECK:         fir.store %{{.*}} to %[[LHS_ADDR_S]]
! CHECK:         omp.yield(%[[ARG0_S]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:       } cleanup {
! CHECK:         fir.freemem
! CHECK:         omp.yield

! CHECK-LABEL: omp.declare_reduction @foo : i32
! CHECK:       init {
! CHECK:         %[[C0_BASE:.*]] = arith.constant 0 : i32
! CHECK:         omp.yield(%[[C0_BASE]] : i32)
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[LHS_BASE:.*]]: i32, %[[RHS_BASE:.*]]: i32):
! CHECK:         arith.addi
! CHECK:         omp.yield

! CHECK-LABEL: func.func @_QPtest_udr_allocatable
! CHECK:         omp.wsloop {{.*}} reduction(byref @foo_byref_box_heap_i32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:         omp.wsloop {{.*}} reduction(byref @foo_byref_box_heap_Uxi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         omp.wsloop {{.*}} reduction(byref @foo_byref_box_heap_UxUxi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
