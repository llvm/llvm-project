! LINEAR on an allocatable: data address operand, rebuilt descriptor in body.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPsimd_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         omp.simd linear(%[[ADDR]] : !fir.heap<i32> = %{{.*}} : i32
! CHECK-SAME:    linear_var_types([i32
! CHECK:           omp.loop_nest
! CHECK:             %[[NEW_BOX:.*]] = fir.embox %[[ADDR]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:             fir.store %[[NEW_BOX]] to %[[NEW_DESC:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             %[[PRIV_A:.*]]:2 = hlfir.declare %[[NEW_DESC]] uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:             hlfir.assign %{{.*}} to %[[PRIV_A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             omp.yield
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
! CHECK:         %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         omp.wsloop linear(%[[ADDR]] : !fir.heap<i32> = %{{.*}} : i32
! CHECK:           omp.loop_nest
! CHECK:             %[[NEW_BOX:.*]] = fir.embox %[[ADDR]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:             fir.store %[[NEW_BOX]] to %[[NEW_DESC:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             %[[PRIV_A:.*]]:2 = hlfir.declare %[[NEW_DESC]] uniq_name("_QFdo_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:             hlfir.assign %{{.*}} to %[[PRIV_A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
subroutine do_linear_allocatable()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp parallel do linear(a:2)
  do i = 1, 10
    a = a + 1
  end do
  !$omp end parallel do
end subroutine

! Composite: the linear operand moves to omp.wsloop.
! CHECK-LABEL: func.func @_QPdo_simd_linear_allocatable
! CHECK:         %[[ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:         omp.wsloop linear(%[[ADDR]] : !fir.heap<i64> = %{{.*}} : i64
! CHECK:           omp.simd
! CHECK:             omp.loop_nest
! CHECK:               %[[NEW_BOX:.*]] = fir.embox %[[ADDR]] : (!fir.heap<i64>) -> !fir.box<!fir.heap<i64>>
! CHECK:               fir.store %[[NEW_BOX]] to %[[NEW_DESC:.*]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:               hlfir.declare %[[NEW_DESC]] uniq_name("_QFdo_simd_linear_allocatableEa") fortran_attrs<allocatable>
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

! The original descriptor is used after the loop.
! CHECK-LABEL: func.func @_QPoriginal_used_after_loop
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QForiginal_used_after_loopEa") fortran_attrs<allocatable>
! CHECK:         omp.simd linear
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
