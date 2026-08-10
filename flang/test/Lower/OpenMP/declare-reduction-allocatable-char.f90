! Allocatable CHARACTER UDR lowers via the existing by-ref char path (character
! is not trivial, so it is not wrapped by the boxed-trivial synthesis)

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s

subroutine test_udr_char_allocatable()
  implicit none
  integer :: i
  character(len=3), allocatable :: s

  !$omp declare reduction(cmax : character(len=3) : omp_out = max(omp_out, omp_in)) &
  !$omp&  initializer(omp_priv = '   ')

  allocate(s)
  s = 'aaa'

  !$omp parallel do reduction(cmax : s)
  do i = 1, 3
    s = max(s, 'bbb')
  end do
end subroutine

! CHECK-LABEL: omp.declare_reduction @{{.*}}cmax_byref_c8x3 : !fir.ref<!fir.char<1,3>>
! CHECK-SAME:  attributes {byref_element_type = !fir.char<1,3>}
! CHECK:       init {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<!fir.char<1,3>>, %{{.*}}: !fir.ref<!fir.char<1,3>>):
! CHECK:         hlfir.assign
! CHECK:         omp.yield
! CHECK:       } combiner {
! CHECK:       ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.char<1,3>>, %{{.*}}: !fir.ref<!fir.char<1,3>>):
! CHECK:         hlfir.char_extremum max
! CHECK:         fir.store %{{.*}} to %[[ARG0]]
! CHECK:         omp.yield(%[[ARG0]] : !fir.ref<!fir.char<1,3>>)

! The clause binds the character op with the boxed allocatable operand.
! CHECK-LABEL: func.func @_QPtest_udr_char_allocatable
! CHECK:         omp.wsloop {{.*}} reduction(byref @{{.*}}cmax_byref_c8x3 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>)
