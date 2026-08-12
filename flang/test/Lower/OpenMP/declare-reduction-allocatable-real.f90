! real, allocatable UDR with initializer(omp_priv = omp_orig) must lower without
! the f32 FloatAttr crash the old fabricated-zero fallback caused.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s

subroutine test_udr_real_allocatable()
  implicit none
  integer :: i
  real, allocatable :: a

  !$omp declare reduction (rmax : real : omp_out = max(omp_out, omp_in)) &
  !$omp & initializer (omp_priv = omp_orig)

  allocate(a)
  a = 0.0

  !$omp parallel do reduction(rmax : a)
  do i = 1, 4
    a = max(a, real(i))
  end do
end subroutine

! CHECK-LABEL: omp.declare_reduction @{{.*}}rmax_byref_box_heap_f32 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK-SAME:  attributes {byref_element_type = f32}
! CHECK:       alloc {
! CHECK:         fir.alloca !fir.box<!fir.heap<f32>>
! CHECK:         omp.yield
! CHECK:       } init {
! CHECK:       ^bb0(%[[MOLD:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>):
! The loaded original f32 element must reach the private element store, not be
! replaced by a fabricated constant.
! CHECK:         %[[MBOX:.*]] = fir.load %[[MOLD]]
! CHECK:         %[[MADDR:.*]] = fir.box_addr %[[MBOX]]
! CHECK:         %[[ORIG:.*]] = fir.load %[[MADDR]] : !fir.heap<f32>
! CHECK:         fir.store %[[ORIG]] to %[[OTMP:.*]] : !fir.ref<f32>
! CHECK:         %[[ODECL:.*]]:2 = hlfir.declare %[[OTMP]] {{.*}}uniq_name = "omp_orig"
! CHECK:         %[[PVAL:.*]] = fir.load %[[ODECL]]#0 : !fir.ref<f32>
! CHECK:         %[[PRIV:.*]] = fir.allocmem f32
! CHECK:         fir.store %[[PVAL]] to %[[PRIV]] : !fir.heap<f32>
! CHECK:         omp.yield
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>, %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>):
! CHECK:         %[[LHS_BOX:.*]] = fir.load %[[ARG0]]
! CHECK:         %[[RHS_BOX:.*]] = fir.load %[[ARG1]]
! CHECK:         %[[LHS_ADDR:.*]] = fir.box_addr %[[LHS_BOX]]
! CHECK:         %[[RHS_ADDR:.*]] = fir.box_addr %[[RHS_BOX]]
! CHECK:         fir.load %[[LHS_ADDR]]
! CHECK:         fir.load %[[RHS_ADDR]]
! CHECK:         fir.store %{{.*}} to %[[LHS_ADDR]]
! CHECK:         omp.yield(%[[ARG0]] : !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:       } cleanup {
! CHECK:         fir.freemem
! CHECK:         omp.yield

! CHECK-LABEL: func.func @_QPtest_udr_real_allocatable
! CHECK:         omp.wsloop {{.*}} reduction(byref @{{.*}}rmax_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>)
