! LINEAR on an allocatable uses a `linear` omp.private with a data_address.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: omp.private {type = linear} @_QFsimd_linear_allocatableEa_linear_box_heap_i32 : !fir.box<!fir.heap<i32>> init {
! CHECK:         fir.allocmem i32
! CHECK:       } data_address {
! CHECK:       ^bb0(%[[ARG:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:         %[[BOX:.*]] = fir.load %[[ARG]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         omp.yield(%[[ADDR]] : !fir.heap<i32>)
! CHECK:       } dealloc {
! CHECK:         fir.freemem

! CHECK-LABEL: func.func @_QPsimd_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         omp.simd linear(@_QFsimd_linear_allocatableEa_linear_box_heap_i32 %[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>> = %{{.*}} : i32,
! CHECK-SAME:    linear_var_types([i32
! CHECK:           omp.loop_nest
! CHECK:             %[[PRIV_A:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:             hlfir.assign %{{.*}} to %[[PRIV_A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:        omp.yield
subroutine simd_linear_allocatable()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp simd linear(a)
  do i = 1, 2
    a = 2
  end do
  !$omp end simd
end subroutine

! CHECK-LABEL: func.func @_QPdo_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         omp.parallel
! CHECK:           omp.wsloop linear(@_QFdo_linear_allocatableEa_linear_box_heap_i32 %[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>> = %{{.*}} : i32)
! CHECK:             omp.loop_nest
! CHECK:               hlfir.declare %[[ARG]] uniq_name("_QFdo_linear_allocatableEa") fortran_attrs<allocatable>
subroutine do_linear_allocatable()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp parallel do linear(a:2)
  do i = 1, 10
    a = a + 2
  end do
  !$omp end parallel do
end subroutine

! Composite: the linear clause is on omp.wsloop.
! CHECK-LABEL: func.func @_QPdo_simd_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_simd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         omp.wsloop linear(@_QFdo_simd_linear_allocatableEa_linear_box_heap_i64 %[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i64>>> = %{{.*}} : i64)
! CHECK-NEXT:      omp.simd linear(@_QFdo_simd_linear_allocatableEi_linear_i32
! CHECK:             omp.loop_nest
! CHECK:               hlfir.declare %[[ARG]] uniq_name("_QFdo_simd_linear_allocatableEa") fortran_attrs<allocatable>
subroutine do_simd_linear_allocatable()
  integer(8), allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp do simd linear(a)
  do i = 1, 10
    a = a + 1
  end do
  !$omp end do simd
end subroutine

! An allocatable loop iteration variable.
! CHECK-LABEL: func.func @_QPallocatable_iv
! CHECK:         %[[I:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFallocatable_ivEi") fortran_attrs<allocatable>
! CHECK:         omp.simd linear(@_QFallocatable_ivEi_linear_box_heap_i32 %[[I]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>> = %{{.*}} : i32,
! CHECK:           omp.loop_nest (%[[IV:.*]]) :
! CHECK:             %[[PRIV_I:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFallocatable_ivEi") fortran_attrs<allocatable>
! CHECK:             %[[BOX:.*]] = fir.load %[[PRIV_I]]#0
! CHECK:             %[[ADDR:.*]] = fir.box_addr %[[BOX]]
! CHECK:             hlfir.assign %{{.*}} to %[[ADDR]] : i32, !fir.heap<i32>
subroutine allocatable_iv()
  integer, allocatable :: i
  integer :: seen(3), k
  allocate(i)
  k = 1
  !$omp simd linear(i:2) linear(k)
  do i = 1, 5, 2
    seen(k) = i
    k = k + 1
  end do
  !$omp end simd
end subroutine

! The original is used after the loop.
! CHECK-LABEL: func.func @_QPoriginal_used_after_loop
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QForiginal_used_after_loopEa") fortran_attrs<allocatable>
! CHECK:         omp.simd linear
! CHECK:           omp.yield
! CHECK:         }
! CHECK:         hlfir.assign %{{.*}} to %[[A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
subroutine original_used_after_loop()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp simd linear(a)
  do i = 1, 2
    a = 2
  end do
  !$omp end simd
  a = 5
end subroutine
